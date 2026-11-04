"""Background-aware text: look at what is actually behind each piece of text before it is drawn.

The solver places text from brand rules alone; this stage runs after the background art exists and before the poster is compiled.
For every text element it samples the pixels under the element's box (the art, or the opaque panel / band the text sits on) and
  1. keeps the intended colour when it reads (WCAG ratio at the worst 10% of pixels),
  2. otherwise switches to the brand colour that reads best (dark brand colour or the light one),
  3. only when no colour reads (a busy, mid-tone area) adds one soft backing panel behind that cluster of text.
Accent roles (eyebrows, icons) take the brand colour that actually harmonises with the ground (accent on dark, tertiary on light).
Everything is derived from pixels and brand palette; nothing is per-brief. Idempotent: it removes its own earlier backings first.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from poster_agent.checks.pixels import hex_rgb
from poster_agent.spec import Box, ElementKind, PlacedElement, PosterSpec
from poster_agent.spec.brand import BrandProfile

GRID = 8.0  # the ground raster is the poster at 1/8 scale
NORMAL_MIN, LARGE_MIN = 4.5, 3.0
LARGE_TEXT = 0.035  # a text element this tall (fraction of poster height, by font size) only needs the large-text ratio
OPAQUE_ROLES = {"info_bar", "highlight_box", "panel", "image_card", "logo_plate", "adapt_scrim"}
SCRIM_OPACITY = 0.72
PREFIX = "adapt."
FADE_FRACTION = 0.09  # of the poster height: how far the art dissolves into an opaque edge band


def _lin(a: np.ndarray) -> np.ndarray:
    a = a / 255.0
    return np.where(a <= 0.03928, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def luminance(rgb: np.ndarray) -> np.ndarray:
    lin = _lin(rgb.astype(float))
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def _lum_of(hex_color: str) -> float:
    return float(luminance(np.array(hex_rgb(hex_color), dtype=float)[None, None, :])[0, 0])


def ratio_p10(text_hex: str, ground_lum: np.ndarray) -> float:
    """Contrast of a text colour against a patch, judged at the worst 10% of its pixels (so a bright streak counts)."""
    lt = _lum_of(text_hex)
    hi, lo = np.maximum(ground_lum, lt), np.minimum(ground_lum, lt)
    return float(np.percentile((hi + 0.05) / (lo + 0.05), 10))


def ground_raster(spec: PosterSpec, root: Path) -> np.ndarray | None:
    """RGB raster (1/8 scale) of the art behind the content, or None for plain solid / gradient grounds (already handled by the
    solver's own colour derivation)."""
    bg = spec.background
    if bg.kind in ("solid", "gradient") or not bg.asset_id or bg.asset_id not in spec.assets:
        return None
    c = spec.canvas
    w, h = max(int(c.width / GRID), 1), max(int(c.height / GRID), 1)
    with Image.open(root / spec.assets[bg.asset_id].path) as im:
        im = im.convert("RGB")
        k = max(w / im.width, h / im.height)  # cover-fit, as the compiler draws it
        im = im.resize((max(int(im.width * k) + 1, w), max(int(im.height * k) + 1, h)))
        x0, y0 = (im.width - w) // 2, (im.height - h) // 2
        return np.asarray(im.crop((x0, y0, x0 + w, y0 + h)))


def _patch(raster: np.ndarray, box: Box) -> np.ndarray:
    x0, y0 = max(int(box.x / GRID), 0), max(int(box.y / GRID), 0)
    x1, y1 = min(int(np.ceil(box.right / GRID)), raster.shape[1]), min(int(np.ceil(box.bottom / GRID)), raster.shape[0])
    if x1 <= x0 or y1 <= y0:
        return raster[:1, :1]
    return raster[y0:y1, x0:x1]


def _enclosing_fill(el: PlacedElement, shapes: list[PlacedElement]) -> str | None:
    """Colour of the opaque shape the text sits on (a band, panel, card), if any."""
    cx, cy = el.box.cx, el.box.cy
    for s in shapes:
        fill = s.props.get("fill")
        if (s.role in OPAQUE_ROLES and isinstance(fill, str) and fill.startswith("#") and s.props.get("opacity") in (None, 1, 1.0)
                and s.z < el.z and s.box.x <= cx <= s.box.right and s.box.y <= cy <= s.box.bottom):
            return fill
    return None


def _is_large(el: PlacedElement, canvas_h: float) -> bool:
    return (el.size or 0) >= LARGE_TEXT * canvas_h


def accent_role(brand: BrandProfile, ground_lum: float) -> str | None:
    """The brand colour for accents that reads on this ground: the teal accent on dark, the tertiary (or primary) on light."""
    p = brand.palette
    for name in (("accent",) if ground_lum < 0.25 else ("tertiary", "primary")):
        hexv = p.get(name)
        if hexv:
            lt, lg = _lum_of(hexv), ground_lum
            if (max(lt, lg) + 0.05) / (min(lt, lg) + 0.05) >= 4.5:
                return hexv
    return None


def _edge_fades(spec: PosterSpec, raster: np.ndarray | None) -> tuple[np.ndarray | None, list[str]]:
    """Where an opaque full-width band (a footer or header) meets generated art, dissolve the art into the band colour over its last
    stretch instead of cutting it off along a hard edge. Adds a feathered gradient shape (decorative, behind everything) and blends the
    same fade into `raster`, so the later text-contrast decisions see what will really be on screen."""
    if raster is None:
        return raster, []
    W, H = spec.canvas.width, spec.canvas.height
    notes: list[str] = []
    raster = raster.astype(float).copy()
    for band in [e for e in spec.placed if e.role == "info_bar" and e.kind == ElementKind.shape]:
        fill = band.props.get("fill")
        if not (isinstance(fill, str) and fill.startswith("#") and len(fill) == 7) or band.props.get("opacity") not in (None, 1, 1.0):
            continue
        at_bottom, at_top = band.box.bottom >= H - 1.0, band.box.y <= 1.0
        if band.box.w < 0.98 * W or at_bottom == at_top:
            continue  # a band that does not span the poster, or that is not on an edge, leaves the art alone
        fh = FADE_FRACTION * H
        if at_bottom:
            y0, y1, a0, a1 = max(band.box.y - fh, 0.0), band.box.y, 0.0, 1.0
        else:
            y0, y1, a0, a1 = band.box.bottom, min(band.box.bottom + fh, H), 1.0, 0.0
        if y1 - y0 < 2.0:
            continue
        grad = {"gradient": {"kind": "linear", "start": [W / 2, y0], "end": [W / 2, y1], "stops": [
            {"offset": 0.0, "color": fill, "opacity": a0}, {"offset": 1.0, "color": fill, "opacity": a1}]}}
        spec.placed.append(PlacedElement(id=f"{PREFIX}fade_{band.id}", kind=ElementKind.shape, role="adapt_fade", z=-60,
                                         box=Box(x=0, y=y0, w=W, h=y1 - y0), props={"fill": grad, "decorative": True}))
        r0, r1 = int(y0 / GRID), min(int(np.ceil(y1 / GRID)), raster.shape[0])
        if r1 > r0:
            ramp = np.linspace(a0, a1, r1 - r0)[:, None, None]
            raster[r0:r1] = raster[r0:r1] * (1 - ramp) + np.array(hex_rgb(fill), dtype=float) * ramp
        notes.append(f"art fades into the {'footer' if at_bottom else 'header'} band {band.id} ({fill}) instead of a hard cut")
    return raster, notes


def adapt_to_background(spec: PosterSpec, brand: BrandProfile, root: Path) -> list[str]:
    """Recolour / back the text of `spec` for the real ground. Mutates spec.placed; returns notes of what was changed."""
    spec.placed = [e for e in spec.placed if not e.id.startswith(PREFIX)]  # idempotent: drop earlier backings
    raster = ground_raster(spec, root)
    raster, notes = _edge_fades(spec, raster)
    shapes = [e for e in spec.placed if e.kind == ElementKind.shape]
    texts = [e for e in spec.placed if e.kind == ElementKind.text and not e.props.get("decorative")]
    dark_c, light_c = brand.palette["primary"], brand.palette["surface"]
    ch = spec.canvas.height
    failing: list[PlacedElement] = []

    for el in texts:
        if el.id in spec.overrides and spec.overrides[el.id].color:  # an explicit repair stays
            continue
        enclosed = _enclosing_fill(el, shapes)
        if enclosed is not None:
            patch = np.full((2, 2, 3), hex_rgb(enclosed), dtype=float)
        elif raster is not None:
            patch = _patch(raster, el.box)
        else:
            continue
        glum = luminance(patch)
        need = LARGE_MIN if _is_large(el, ch) else NORMAL_MIN
        current = el.color or dark_c
        if ratio_p10(current, glum) >= need:
            continue
        best = max((dark_c, light_c), key=lambda c: ratio_p10(c, glum))
        if ratio_p10(best, glum) >= need:
            notes.append(f"{el.id}: {current} -> {best} (readable on the pixels behind it)")
            el.color = best
        else:
            failing.append(el)  # no colour reads here: needs a backing

    # soft gradient bands behind clusters of unreadable text: full width, feathered top and bottom, so there is no box edge at all
    failing.sort(key=lambda e: e.box.y)
    clusters: list[list[PlacedElement]] = []
    for el in failing:
        if clusters and el.box.y - max(m.box.bottom for m in clusters[-1]) <= 0.05 * ch:  # vertical neighbours share one band
            clusters[-1].append(el)
        else:
            clusters.append([el])
    W = spec.canvas.width
    for i, group in enumerate(clusters):
        top_y, bottom_y = min(m.box.y for m in group), max(m.box.bottom for m in group)
        union = Box(x=0, y=top_y, w=W, h=bottom_y - top_y)
        patch = _patch(raster, union) if raster is not None else np.full((2, 2, 3), 128.0)
        light_ground = float(np.mean(luminance(patch))) >= 0.35
        panel, ink = (light_c, dark_c) if light_ground else (dark_c, light_c)
        for m in group:
            m.color = ink
        feather = 0.055 * ch
        y0, y1 = max(top_y - feather - 0.01 * ch, 0.0), min(bottom_y + feather + 0.01 * ch, ch)
        stops = [{"offset": 0.0, "color": panel, "opacity": 0.0}, {"offset": 0.28, "color": panel, "opacity": SCRIM_OPACITY},
                 {"offset": 0.72, "color": panel, "opacity": SCRIM_OPACITY}, {"offset": 1.0, "color": panel, "opacity": 0.0}]
        grad = {"gradient": {"kind": "linear", "start": [W / 2, y0], "end": [W / 2, y1], "stops": stops}}
        spec.placed.append(PlacedElement(id=f"{PREFIX}scrim{i}", kind=ElementKind.shape, role="adapt_scrim", z=min(m.z for m in group) - 1,
                                         box=Box(x=0, y=y0, w=W, h=y1 - y0), props={"fill": grad, "decorative": True}))
        notes.append(f"{', '.join(m.id for m in group[:3])}{'...' if len(group) > 3 else ''}: a soft "
                     f"{'light' if light_ground else 'dark'} gradient band behind them (no colour reads on this part of the art)")

    # accent roles: the brand colour that harmonises with this ground
    glob_lum = float(np.mean(luminance(raster))) if raster is not None else None
    if glob_lum is not None:
        for el in texts:
            if el.role == "label" and not (el.id in spec.overrides and spec.overrides[el.id].color):
                a = accent_role(brand, float(np.mean(luminance(_patch(raster, el.box)))))
                if a and a != el.color:
                    notes.append(f"{el.id}: accent colour {a} (harmonises with the ground)")
                    el.color = a
        for el in spec.placed:
            if el.role == "icon" and "recolor" in el.props:
                a = accent_role(brand, float(np.mean(luminance(_patch(raster, el.box)))))
                if a:
                    el.props["recolor"] = a
    return notes
