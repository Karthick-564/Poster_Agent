"""The LOOK of a poster: what makes two posters on the same skeleton look different.

The institution's own posters vary in the ground (white, solid brand blue, a soft tint, dark art), in how the headline is set (colour or
weight contrast between its lines), and in what the imagery does (a cutout on a clean ground, OR art as the ground, never both). A look
fixes those choices. It is chosen per brief (deterministically from the genre and the title, or forced with brief.look) and APPLIED to
the plan after the planner has answered: the ground, the type style and the image idea ONLY. Everything else (header, logo, footer, text
colours, title treatment, structure) is the planner's own decision and is never rewritten here. Nothing here writes copy."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from poster_agent.plan.schema import PlanBackground, PlanOutput, PlanStop
from poster_agent.spec import ComponentSpec
from poster_agent.spec.brand import BrandProfile


@dataclass(frozen=True)
class Look:
    name: str
    about: str
    ground: str  # surface | primary | tint | generated
    type_style: str
    title_style: str  # accent_last | accent_first | weight_split | highlight_last
    imagery: str  # cutout: a generated or supplied cutout on the plain ground; art: generated art IS the ground (no cutout)
    dark: bool = False  # light text on this ground


LOOKS: dict[str, Look] = {
    "clean_light": Look("clean_light", "white ground, navy headline with a light-weight last line, one cutout or photo anchoring the bottom",
                        "surface", "editorial", "weight_split", "cutout"),
    "brand_blue": Look("brand_blue", "solid brand-blue ground, white headline with an accent last line, a cutout standing on the bottom edge",
                       "primary", "bold", "accent_last", "cutout", dark=True),
    "tint_soft": Look("tint_soft", "soft light-grey tint ground, navy headline with a solid bar behind its last line, a cutout or photo",
                      "tint", "elegant", "highlight_last", "cutout"),
    "dark_art": Look("dark_art", "full-bleed generated key visual as the ground (a cutout may stand on it), brand-blue header band, light headline",
                     "generated", "condensed", "accent_last", "art", dark=True),
}

GENRE_LOOKS: dict[str, list[str]] = {
    "workshop": ["clean_light", "brand_blue"], "recruitment": ["brand_blue", "tint_soft"], "admission": ["clean_light", "tint_soft"],
    "conference": ["clean_light", "dark_art"], "talk": ["tint_soft", "clean_light"], "cultural": ["dark_art", "brand_blue"],
    "sports": ["dark_art", "brand_blue"], "commemoration": ["clean_light", "tint_soft"], "placement": ["brand_blue", "clean_light"],
    "launch": ["dark_art", "tint_soft"],
}


def look_for(brief) -> Look:
    """The look of this brief: forced by brief.look, else from the genre's looks, chosen by the title (stable for a given title)."""
    from poster_agent.skills import genre_of
    forced = getattr(brief, "look", None)
    if forced in LOOKS:
        return LOOKS[forced]
    names = GENRE_LOOKS.get(genre_of(brief)) or ["clean_light", "brand_blue", "tint_soft"]
    h = int(hashlib.sha1(("look|" + brief.event_title).encode()).hexdigest(), 16)
    return LOOKS[names[h % len(names)]]


def tint_hex(brand: BrandProfile) -> str:
    """A very light wash of the brand's primary colour over white (the soft ground)."""
    p = brand.palette["primary"].lstrip("#")
    r, g, b = (int(p[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: int(255 * 0.93 + c * 0.07)  # a faint wash of the brand colour, not a grey
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def _mix(a: str, b: str, t: float) -> str:
    """Hex colour a moved toward b by t."""
    a, b = a.lstrip("#"), b.lstrip("#")
    return "#" + "".join(f"{int(int(a[i:i + 2], 16) * (1 - t) + int(b[i:i + 2], 16) * t):02x}" for i in (0, 2, 4))


def ground_background(look: Look, brand: BrandProfile) -> PlanBackground:
    """The ground of a plain look: a soft vertical gradient (depth, no ornament) instead of a flat colour."""
    p, white, tint = brand.palette["primary"], "#ffffff", tint_hex(brand)
    top, bottom = {"surface": (white, tint), "tint": (white, tint),
                   "primary": (_mix(p, white, 0.16), p)}[look.ground]
    # VectorCraft runs a 90 degree gradient from the BOTTOM edge to the top: stop 0 is the bottom colour, stop 1 the top colour
    return PlanBackground(kind="gradient", color=bottom, angle=90.0,
                          gradient=[PlanStop(offset=0.0, color=bottom), PlanStop(offset=1.0, color=top)])


def _walk(items: list[ComponentSpec]):
    for c in items:
        yield c
        yield from _walk(c.children)


def _is_generated_image(c: ComponentSpec, assets: set[str]) -> bool:
    return c.component in ("HeroImage", "BottomCutout") and c.content.get("asset") in assets


def apply_look(plan: PlanOutput, brand: BrandProfile, look: Look) -> list[str]:
    """Impose `look` on the plan in place; returns what was changed (for the run notes)."""
    notes: list[str] = []
    generated_heroes = {a.id for a in plan.generated_assets}
    had_art = plan.background.kind == "generated" and plan.background.art is not None
    has_cutout = any(_is_generated_image(c, generated_heroes) for c in _walk(plan.components))
    effective = look
    if look.ground == "generated" and not had_art:  # the planner offered no art to be the ground: use the closest plain look
        effective = LOOKS["brand_blue"]
        notes.append(f"look {look.name} needs generated art and the plan has none: used {effective.name}")
    elif look.ground != "generated" and had_art and has_cutout:  # the planner asked for both: keep the art, it is the key visual
        effective = LOOKS["dark_art"]
        notes.append(f"look {look.name} -> {effective.name}: the plan has background art and a cutout, so the art is kept as the ground")
    plan.look = effective.name

    # ---- ground: a plain look has a plain ground; an art look keeps the art at full strength ----
    if effective.ground == "generated":
        plan.background.art.tone = "dark"
    else:
        if had_art:
            notes.append("background art dropped: a plain look keeps a plain ground")
        panel = plan.background.panel  # the planner's two-tone panel is its own decision: only the ground under it is the look's
        plan.background = ground_background(effective, brand)
        plan.background.panel = panel
    plan.type_style = effective.type_style

    # ---- imagery: an art look keeps a cutout (the art is the setting, the cutout the subject); plain looks tune the hero's light ----
    if effective.imagery == "art":
        pass
    else:
        for a in plan.generated_assets:  # the hero keeps its natural colours; only the LIGHT is matched to this ground
            if "natural colours" not in a.subject:
                a.subject = (a.subject.rstrip(". ") + ", true natural colours, soft diffuse daylight from the upper left, matte finish, "
                             "minimal reflections, calm and understated")

    # ---- ornament dividers and edge stripes are not part of the institution's look ----
    before = len(plan.components)
    plan.components = [c for c in plan.components if not (c.component == "Divider" and c.variant == "ornament")]
    if len(plan.components) < before:
        notes.append("ornament divider removed")
    kept = [d for d in plan.decorations if d.kind in ("border", "frame")]
    if len(kept) < len(plan.decorations):
        notes.append("decorative stripes removed (only a border or frame is kept)")
    plan.decorations = kept

    notes.append(f"look {effective.name}: {effective.about}")
    return notes
