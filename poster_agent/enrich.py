"""Deterministic enrichment: give plain posters texture and rhythm with brand-native decorations (no LLM, no GPU).

Two entry points:
  default_preset  — applied before the first render when a poster has a plain background and the planner chose no decorations
  top_up          — applied by the refine loop to the largest dead regions the emptiness check measured
Decorations sit behind all content (see layout/decor.py), so enrichment can never cover text, logos or photos.
"""
from __future__ import annotations

from poster_agent.checks.pixels import hex_rgb, rel_luminance
from poster_agent.layout.base import resolve_color
from poster_agent.spec import Box, Decoration, ElementKind, PosterSpec
from poster_agent.spec.brand import BrandProfile

MIN_FREE_BAND_FRACTION = 0.07  # a text-free horizontal band must be at least this tall (fraction of the poster) to take a pattern
TOP_UP_SEQUENCE = ["dot_grid", "rings", "glow", "diagonal_band"]  # tried in order on successive enrich rounds


def is_dark_ground(spec: PosterSpec, brand: BrandProfile, art_tone: str | None = None) -> bool:
    if art_tone:
        return art_tone == "dark"
    bg = spec.background
    if bg.kind == "gradient" and bg.gradient:
        lums = [rel_luminance(hex_rgb(resolve_color(brand, s.color))) for s in bg.gradient]
        return sum(lums) / len(lums) < 0.25
    return rel_luminance(hex_rgb(resolve_color(brand, bg.color))) < 0.25


def free_bands(spec: PosterSpec) -> list[Box]:
    """Full-width horizontal bands with no text in them, tall enough to hold a pattern, tallest first."""
    c = spec.canvas
    spans = sorted((e.box.y, e.box.bottom) for e in spec.placed if e.kind == ElementKind.text and not e.props.get("decorative"))
    free, cur = [], 0.0
    for y0, y1 in spans + [(c.height, c.height)]:
        if y0 - cur >= MIN_FREE_BAND_FRACTION * c.height:
            free.append(Box(x=0, y=cur, w=c.width, h=y0 - cur))
        cur = max(cur, y1)
    return sorted(free, key=lambda b: -b.h)


def _used(spec: PosterSpec) -> set[str]:
    return {d.kind for d in spec.decorations}


def hero_glow(spec: PosterSpec, brand: BrandProfile, dark: bool) -> Decoration | None:
    """A pool of light behind the hero: gives it a setting and lifts it off the ground. None if there is no hero or no glow allowed."""
    hero = next((e for e in spec.placed if e.role == "hero" and e.kind == ElementKind.image), None)
    if hero is None or "glow" not in brand.allowed_decorations:
        return None
    pad = 0.25 * max(hero.box.w, hero.box.h)
    return Decoration(id="auto_glow", kind="glow", region=hero.box.expand(pad), color="accent", opacity=0.30 if dark else 0.16, source="auto")


def default_preset(spec: PosterSpec, brand: BrandProfile, dark: bool) -> list[Decoration]:
    """A clean, brand-coloured base: a bold brand border (or a top stripe where borders are not allowed). Ornaments (glow, dot fields, corner shapes) are added only
    when the brand's allow-list opts in; the default brand does not allow them, so plain space is solved by layout, not decoration."""
    ok = set(brand.allowed_decorations)
    main, accent = ("surface", "accent") if dark else ("primary", "accent")
    out: list[Decoration] = []
    if "border" in ok:
        out.append(Decoration(id="auto_border", kind="border", color="accent" if dark else "primary", opacity=1.0, source="auto"))
    if "corner_shapes" in ok:
        out.append(Decoration(id="auto_corners", kind="corner_shapes", color=main, opacity=0.10 if dark else 0.06, source="auto"))
    glow = hero_glow(spec, brand, dark)
    if glow is not None:
        out.append(glow)
    bands = free_bands(spec)
    if "dot_grid" in ok and bands:
        out.append(Decoration(id="auto_dots", kind="dot_grid", region=bands[0], color=main if dark else accent,
                              opacity=0.10 if dark else 0.55, source="auto"))
    if "stripe" in ok and "border" not in ok:
        out.append(Decoration(id="auto_stripe", kind="stripe", region="top", color=accent, opacity=0.9, source="auto"))
    return out


def top_up(spec: PosterSpec, brand: BrandProfile, regions: list[Box], dark: bool, step: int) -> list[str]:
    """Add one decoration for each of the two largest dead regions. `step` selects which kind is tried (a different one each round)."""
    ok = [k for k in TOP_UP_SEQUENCE if k in brand.allowed_decorations]
    if not ok:
        return []
    actions: list[str] = []
    main = "surface" if dark else "primary"
    for i, region in enumerate(regions[:2]):
        kind = ok[(step + i) % len(ok)]
        n = len([d for d in spec.decorations if d.id.startswith("auto_top")]) + 1
        # top-ups must be visible (and register in the emptiness measure): faint 10-12% marks on a dark ground do neither
        if kind == "glow":
            color, opacity = ("accent", 0.30 if dark else 0.18)
        elif kind == "diagonal_band":
            color, opacity = (("surface", 0.14) if dark else ("primary", 0.08))
        else:  # dot_grid, rings
            color, opacity = (("surface", 0.28) if dark else ("accent", 0.5))
        spec.decorations.append(Decoration(id=f"auto_top{n}_{kind}", kind=kind, region=region, color=color, opacity=opacity, source="auto"))
        actions.append(f"added {kind} decoration over a {region.w:.0f}x{region.h:.0f}pt dead region at ({region.x:.0f},{region.y:.0f})")
    return actions


def watermark_candidates(texts: list[str]) -> list[str]:
    """Verbatim short strings from the brief that suit a giant faint watermark (a year first, then the first word of the title)."""
    import re
    out: list[str] = []
    for t in texts:
        m = re.search(r"\b(19|20)\d{2}\b", t)
        if m and m.group(0) not in out:
            out.append(m.group(0))
    return out
