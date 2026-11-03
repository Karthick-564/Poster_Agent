"""Decorations: brand-native vector ornament drawn between the background and the content.

Each Decoration expands into ordinary PlacedElements (shapes, one optional text) with negative z, so the compiler draws them
after the background and before everything else, and they can never cover text, logos or photos. They are marked
`props["decorative"] = True`, which tells the checks not to hold them to legibility/contrast/overlap rules.
Decorations are the answer to plain space: they give empty areas texture and rhythm without inventing content.
"""
from __future__ import annotations

import math

from poster_agent.spec import Box, CanvasSpec, Decoration, ElementKind, PlacedElement
from poster_agent.spec.brand import BrandProfile

from .base import resolve_color
from .fonts import TextMeasurer

MAX_DOTS = 420
Z = {"watermark_text": -59, "glow": -58, "corner_shapes": -57, "diagonal_band": -56, "dot_grid": -55, "rings": -54,
     "frame": -53, "stripe": -52, "border": -51}


def region_box(region: Box | str, c: CanvasSpec) -> Box:
    if isinstance(region, Box):
        return region
    w, h = c.width, c.height
    return {"full": Box(x=0, y=0, w=w, h=h), "top": Box(x=0, y=0, w=w, h=0.35 * h), "bottom": Box(x=0, y=0.65 * h, w=w, h=0.35 * h),
            "left": Box(x=0, y=0, w=0.4 * w, h=h), "right": Box(x=0.6 * w, y=0, w=0.4 * w, h=h),
            "center": Box(x=0.2 * w, y=0.25 * h, w=0.6 * w, h=0.5 * h)}[region]


def _el(d: Decoration, n: int, box: Box, **props) -> PlacedElement:
    return PlacedElement(id=f"decor.{d.id}.{n}", kind=ElementKind.shape, role=f"decor_{d.kind}", box=box, z=Z[d.kind],
                         props={"decorative": True, **props})


def _rect_corners(b: Box, deg: float) -> list[tuple[float, float]]:
    a = math.radians(deg)
    out = []
    for dx, dy in ((-b.w / 2, -b.h / 2), (b.w / 2, -b.h / 2), (b.w / 2, b.h / 2), (-b.w / 2, b.h / 2)):
        out.append((b.cx + dx * math.cos(a) - dy * math.sin(a), b.cy + dx * math.sin(a) + dy * math.cos(a)))
    return out


def hits_zone(el: PlacedElement, z: Box) -> bool:
    """Does this decoration element touch the keep-out zone `z`? Conservative (may say yes when it barely misses)."""
    b, p = el.box, el.props
    if p.get("shape") == "ellipse":
        r = max(b.w, b.h) / 2
        nx, ny = min(max(b.cx, z.x), z.right), min(max(b.cy, z.y), z.bottom)  # nearest point of z to the centre
        d_near = math.hypot(b.cx - nx, b.cy - ny)
        if p.get("fill") is None:  # ring: only its outline matters
            d_far = max(math.hypot(b.cx - cx, b.cy - cy) for cx in (z.x, z.right) for cy in (z.y, z.bottom))
            return d_near <= r <= d_far
        return d_near <= r
    if p.get("rotation"):
        pts = _rect_corners(b, p["rotation"])
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        return Box(x=min(xs), y=min(ys), w=max(xs) - min(xs), h=max(ys) - min(ys)).intersects(z)
    if p.get("fill") is None and p.get("stroke"):  # frame: only the outline matters
        t = max(float(p.get("stroke_width", 2.0)) * 2, 4.0)
        return b.intersects(z) and not b.inset(t).contains(z)
    return b.intersects(z)


def expand(d: Decoration, canvas: CanvasSpec, brand: BrandProfile, measurer: TextMeasurer,
           keepout: list[Box] | None = None) -> list[PlacedElement]:
    els = _expand(d, canvas, brand, measurer)
    if not keepout:
        return els
    out = []
    for el in els:
        box = el.box
        if el.kind == ElementKind.text:  # a watermark only occupies its glyph band, not its whole frame
            box = Box(x=el.box.x, y=el.box.cy - (el.size or 0) * 0.6, w=el.box.w, h=(el.size or 0) * 1.2)
            probe = el.model_copy(update={"box": box})
        else:
            probe = el
        if not any(hits_zone(probe, z) for z in keepout):
            out.append(el)
    return out


def _expand(d: Decoration, canvas: CanvasSpec, brand: BrandProfile, measurer: TextMeasurer) -> list[PlacedElement]:
    col = resolve_color(brand, d.color)
    R = region_box(d.region, canvas)
    W, H = canvas.width, canvas.height
    short = min(W, H)
    out: list[PlacedElement] = []

    if d.kind == "glow":  # a soft pool of light/colour behind the content
        w, h = R.w * 1.25, R.h * 1.25
        out.append(_el(d, 0, Box(x=R.cx - w / 2, y=R.cy - h / 2, w=w, h=h), shape="ellipse", fill={"gradient": {
            "kind": "radial", "stops": [{"offset": 0.0, "color": col, "opacity": d.opacity}, {"offset": 1.0, "color": col, "opacity": 0.0}]}}))

    elif d.kind == "dot_grid":  # halftone: dots swell toward the middle of the region and thin out at its edges
        spacing = 0.026 * W / d.density
        while (R.w / spacing) * (R.h / spacing) > MAX_DOTS:
            spacing *= 1.12
        cols, rows = max(int(R.w / spacing), 1), max(int(R.h / spacing), 1)
        x0, y0 = R.x + (R.w - cols * spacing) / 2, R.y + (R.h - rows * spacing) / 2
        half = math.hypot(R.w, R.h) / 2
        n = 0
        for r in range(rows):
            for c in range(cols):
                cx, cy = x0 + (c + 0.5) * spacing, y0 + (r + 0.5) * spacing
                k = 1.0 - 0.65 * min(math.hypot(cx - R.cx, cy - R.cy) / half, 1.0)
                dia = spacing * 0.34 * k
                out.append(_el(d, n, Box(x=cx - dia / 2, y=cy - dia / 2, w=dia, h=dia), shape="ellipse", fill=col, opacity=d.opacity))
                n += 1

    elif d.kind == "rings":  # concentric ripples centred in the region (may run off the canvas)
        count = max(3, min(round(5 * d.density), 9))
        for i in range(count):
            r = 0.10 * short + i * 0.085 * short
            out.append(_el(d, i, Box(x=R.cx - r, y=R.cy - r, w=2 * r, h=2 * r), shape="ellipse", fill=None, stroke=col,
                           stroke_width=max(1.0, 0.0016 * W), opacity=d.opacity))

    elif d.kind == "corner_shapes":  # two large circles bleeding off opposite corners
        for i, (cx, cy, r) in enumerate(((0.0, 0.0, 0.30 * W), (W, H, 0.24 * W))):
            out.append(_el(d, i, Box(x=cx - r, y=cy - r, w=2 * r, h=2 * r), shape="ellipse", fill=col, opacity=d.opacity))

    elif d.kind == "diagonal_band":  # a slanted colour band across the region
        length = 1.5 * math.hypot(R.w, R.h)
        thick = 0.055 * max(W, H)
        out.append(_el(d, 0, Box(x=R.cx - length / 2, y=R.cy - thick / 2, w=length, h=thick), fill=col, opacity=d.opacity, rotation=-18))
        out.append(_el(d, 1, Box(x=R.cx - length / 2, y=R.cy + thick * 1.1, w=length, h=thick * 0.28), fill=col, opacity=min(d.opacity * 1.6, 1.0),
                       rotation=-18))

    elif d.kind == "frame":  # a fine inset keyline around the poster
        m = 0.022 * short
        out.append(_el(d, 0, Box(x=m, y=m, w=W - 2 * m, h=H - 2 * m), fill=None, stroke=col, stroke_width=max(1.5, 0.002 * W),
                       opacity=min(d.opacity * 2.5, 1.0)))

    elif d.kind == "border":  # a bold border on the poster edge plus a fine inner keyline (a designer's double border)
        t = 0.014 * short
        out.append(_el(d, 0, Box(x=t / 2, y=t / 2, w=W - t, h=H - t), fill=None, stroke=col, stroke_width=t, opacity=d.opacity))
        k = 2.1 * t
        out.append(_el(d, 1, Box(x=k, y=k, w=W - 2 * k, h=H - 2 * k), fill=None, stroke=col, stroke_width=max(1.2, 0.0016 * W),
                       opacity=min(d.opacity * 0.6, 1.0)))

    elif d.kind == "stripe":  # a solid bar on the edge of the region (top/bottom: horizontal, left/right: vertical)
        t = 0.012 * short
        if isinstance(d.region, str) and d.region in ("left", "right"):
            x = R.x if d.region == "left" else R.right - t
            out.append(_el(d, 0, Box(x=x, y=0, w=t, h=H), fill=col, opacity=d.opacity))
        else:
            y = R.y if (isinstance(d.region, str) and d.region != "bottom") or not isinstance(d.region, str) else R.bottom - t
            out.append(_el(d, 0, Box(x=0, y=y, w=W, h=t), fill=col, opacity=d.opacity))

    elif d.kind == "watermark_text":  # huge faint lettering (verbatim brief text, e.g. the year) behind the content
        if d.text:
            font = brand.fonts.get("display_heavy") or brand.fonts["display"]
            w100 = max(measurer.width(d.text, font.family, font.style, 100.0), 1.0)
            size = min(100.0 * 0.92 * R.w / w100, R.h * 0.95)
            h = size * 1.25
            out.append(PlacedElement(
                id=f"decor.{d.id}.0", kind=ElementKind.text, role="decor_watermark_text", z=Z[d.kind], text=d.text, font=font.family,
                style=font.style, size=round(size, 2), color=col, align="center", box=Box(x=R.x, y=R.cy - h / 2, w=R.w, h=h),
                props={"decorative": True, "opacity": d.opacity}))
    return out


def expand_all(decorations: list[Decoration], canvas: CanvasSpec, brand: BrandProfile, measurer: TextMeasurer,
               keepout: list[Box] | None = None) -> list[PlacedElement]:
    out: list[PlacedElement] = []
    for d in decorations:
        out += expand(d, canvas, brand, measurer, keepout)
    return out
