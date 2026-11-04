"""The LOOK of a poster: what makes two posters on the same skeleton look different.

The institution's own posters vary in the ground (white, solid brand blue, a soft tint, dark art), in how the headline is set (colour or
weight contrast between its lines), and in what the imagery does (a cutout on a clean ground, OR art as the ground, never both). A look
fixes those choices. It is chosen per brief (deterministically from the genre and the title, or forced with brief.look) and APPLIED to
the plan after the planner has answered, so it cannot be skipped: ground, type style, a title treatment, consistent text colours, a header
band when the ground is art, and one image idea. Nothing here writes copy; it only changes colours, fonts, backgrounds and styling."""
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
    header_band: str | None = None  # palette name of a band behind the header strip when the ground is art
    dark: bool = False  # light text on this ground


LOOKS: dict[str, Look] = {
    "clean_light": Look("clean_light", "white ground, navy headline with a light-weight last line, one cutout or photo anchoring the bottom",
                        "surface", "editorial", "weight_split", "cutout"),
    "brand_blue": Look("brand_blue", "solid brand-blue ground, white headline with an accent last line, a cutout standing on the bottom edge",
                       "primary", "bold", "accent_last", "cutout", dark=True),
    "tint_soft": Look("tint_soft", "soft light-grey tint ground, navy headline with a solid bar behind its last line, a cutout or photo",
                      "tint", "elegant", "highlight_last", "cutout"),
    "dark_art": Look("dark_art", "generated art is the ground (no cutout), a brand-blue header band the logo merges into, light headline",
                     "generated", "condensed", "accent_last", "art", header_band="primary", dark=True),
}

GENRE_LOOKS: dict[str, list[str]] = {
    "workshop": ["clean_light", "brand_blue"], "recruitment": ["brand_blue", "tint_soft"], "admission": ["clean_light", "tint_soft"],
    "conference": ["clean_light", "dark_art"], "talk": ["tint_soft", "clean_light"], "cultural": ["dark_art", "brand_blue"],
    "sports": ["dark_art", "brand_blue"], "commemoration": ["clean_light", "tint_soft"], "placement": ["brand_blue", "clean_light"],
    "launch": ["dark_art", "tint_soft"],
}

LIGHT_ON_DARK = {"text", "primary", "muted", "on_surface", "neutral"}
DARK_ON_LIGHT = {"surface", "on_primary", "white"}
COLOR_KEYS = {"TitleStack": ["title_color", "subtitle_color", "tagline_color", "eyebrow_color"], "HeaderRow": ["color"], "FactLine": ["color"],
              "TextBlock": ["color"], "IconRow": ["label_color", "icon_color"], "FactStrip": ["value_color", "label_color"],
              "QRBlock": ["caption_color"], "BigWord": ["color"], "BigTitle": ["color"]}
KEEP_ACCENT = {"accent", "tertiary"}


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
    effective = look
    if look.ground == "generated" and not had_art:  # the planner offered no art to be the ground: use the closest plain look
        effective = LOOKS["brand_blue"]
        notes.append(f"look {look.name} needs generated art and the plan has none: used {effective.name}")
    plan.look = effective.name

    # ---- ground: plain looks never keep generated background art (one image idea) ----
    if effective.ground == "generated":
        plan.background.art.tone = "dark"
    else:
        if had_art:
            notes.append("background art dropped: a cutout look keeps a plain ground")
        plan.background = ground_background(effective, brand)
    plan.type_style = effective.type_style

    # ---- imagery: art looks carry no cutout; cutout looks keep it ----
    if effective.imagery == "art":
        drop = {c.id for c in _walk(plan.components) if _is_generated_image(c, generated_heroes)}
        if drop:
            plan.components = [c for c in plan.components if c.id not in drop]
            used = {c.content.get("asset") for c in _walk(plan.components)}
            plan.generated_assets = [a for a in plan.generated_assets if a.id in used]
            notes.append("cutout dropped: the art is the image of this look")
    else:
        for a in plan.generated_assets:  # the hero keeps its natural colours; only the LIGHT is matched to this ground
            if "natural colours" not in a.subject:
                a.subject = (a.subject.rstrip(". ") + ", true natural colours, soft diffuse daylight from the upper left, matte finish, "
                             "minimal reflections, calm and understated")

    # ---- a header or logo strip is placed inside the safe margins, never as a full-bleed edge band ----
    for c in plan.components:
        if c.component in ("HeaderRow", "LogoBlock", "LogoStrip") and c.anchor != "flow":
            c.anchor = "flow"
            notes.append(f"{c.id}: a header / logo strip is placed inside the margins (anchor set to flow)")

    # ---- the header is the full logo alone; its date / time / venue lines belong to the footer ----
    for i, c in enumerate(plan.components):
        if c.component == "HeaderRow" and c.anchor == "flow":
            lines = [t for t in c.content.get("lines", []) if t]
            logo = {k: v for k, v in (("logo", c.content.get("logo")), ("width_pct", c.content.get("logo_pct")),
                                      ("band", c.content.get("band"))) if v}
            plan.components[i] = ComponentSpec(id=c.id, component="LogoBlock", emphasis=c.emphasis, content=logo)
            if lines:
                foot = next((f for f in plan.components if f.component == "InfoBar" and f.anchor == "bottom"), None)
                if foot is None:
                    foot = ComponentSpec(id="footer", component="InfoBar", variant="footer", anchor="bottom", emphasis="secondary",
                                         content={"items": []})
                    plan.components.append(foot)
                foot.content["items"] = lines + [t for t in foot.content.get("items", []) if t not in lines]
                notes.append("date / time / venue moved from the header to the footer")
            break

    # ---- one standard footer on every poster ----
    for c in plan.components:
        if c.component == "InfoBar" and c.anchor == "bottom" and c.variant in ("band", "footer"):
            c.variant = "footer"
            c.content["fill"], c.content["text_color"] = "primary", "surface"
            c.content.pop("fill_to", None)

    # ---- ornament dividers and edge stripes are not part of the institution's look ----
    before = len(plan.components)
    plan.components = [c for c in plan.components if not (c.component == "Divider" and c.variant == "ornament")]
    if len(plan.components) < before:
        notes.append("ornament divider removed")
    kept = [d for d in plan.decorations if d.kind in ("border", "frame")]
    if len(kept) < len(plan.decorations):
        notes.append("decorative stripes removed (only a border or frame is kept)")
    plan.decorations = kept

    # ---- text colours follow the ground ----
    dark = effective.dark
    for c in _walk(plan.components):
        if c.anchor != "flow":
            continue
        for key in COLOR_KEYS.get(c.component, []):
            v = c.content.get(key)
            if v is None:
                continue
            if dark and v in LIGHT_ON_DARK:
                c.content[key] = "surface"
            elif not dark and v in DARK_ON_LIGHT:
                c.content[key] = "primary" if key != "subtitle_color" else "text"
        if c.component == "TitleStack":
            c.content.setdefault("title_color", "surface" if dark else "primary")
            c.content.setdefault("subtitle_color", "surface" if dark else "text")
            if len((c.content.get("title") or "").split()) >= 2:  # a title treatment is part of the look, never optional
                c.content["title_style"] = effective.title_style
                if effective.title_style == "accent_last" and not dark:
                    c.content["title_style"] = "weight_split"

    # ---- header: on art, the header strip sits on a brand band the logo merges into ----
    flow = [c for c in plan.components if c.anchor == "flow"]
    if effective.header_band and flow and flow[0].component in ("HeaderRow", "LogoBlock", "LogoStrip"):
        flow[0].content["band"] = effective.header_band
        if flow[0].component == "HeaderRow":
            flow[0].content["color"] = "surface"
    notes.append(f"look {effective.name}: {effective.about}")
    return notes
