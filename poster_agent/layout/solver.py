"""Deterministic layout: ComponentSpecs -> PlacedElements. No LLM, no per-brief rules (guardrail 5).

Flow components stack top-to-bottom inside the safe box; 'top'/'bottom' components are full-bleed bands pinned to the
canvas edges. If content does not fit, type scales down globally (never below the role floors); if it still does not
fit the solver reports a typed `fit` Issue instead of silently overflowing (the planner decides what to drop).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from poster_agent.spec import Box, ComponentSpec, ElementKind, Issue, IssueCategory, PlacedElement, PosterSpec
from poster_agent.spec.brand import BrandProfile
from poster_agent.spec.formats import MM_TO_PT

from . import components, skeleton, structure  # noqa: F401  (registers components)
from .decor import expand_all
from .base import REGISTRY, Ctx
from .fonts import TextMeasurer
from .zones import template as get_template

PIN_TOP = {"HeaderRow", "LogoBlock", "LogoStrip"}
BASE_GAP = 0.022  # fraction of canvas height between stacked components
EMPHASIS_GAP = {"hero": 1.6, "primary": 1.0, "secondary": 0.8, "tertiary": 0.6}
SCALES = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
UP_SCALES = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3]  # when there is room to spare, grow type/hero instead of widening gaps
MIN_SLACK_FOR_GROWTH = 0.10  # only if at least 10% of the available height is unused at scale 1.0
MAX_GAP_GROWTH = 1.35


def _gap(ctx: Ctx, spec: ComponentSpec, gap_scale: float = 1.0) -> float:
    return BASE_GAP * ctx.canvas.height * ctx.scale * gap_scale * EMPHASIS_GAP[spec.emphasis]


def _component(spec: ComponentSpec):
    try:
        return REGISTRY[spec.component]
    except KeyError:
        raise KeyError(f"unknown component {spec.component!r}; known: {sorted(REGISTRY)}") from None


def stack(specs: list[ComponentSpec], ctx: Ctx, box: Box, gap_scale: float = 1.0) -> tuple[list[PlacedElement], float]:
    """Stack components top-down in `box` (box.h ignored); returns (elements, total height)."""
    els: list[PlacedElement] = []
    y = box.y
    for i, spec in enumerate(specs):
        comp = _component(spec)
        e, h = comp.build(spec, ctx, box.x, y, box.w)
        els += e
        y += h + (_gap(ctx, specs[i + 1], gap_scale) if i + 1 < len(specs) else 0.0)
    return els, y - box.y


@dataclass
class LayoutResult:
    placed: list[PlacedElement]
    assets: dict
    issues: list[Issue]
    scale: float
    fits: bool
    flow_box: Box | None = None
    bleed_ids: set[str] = field(default_factory=set)


def _ground(spec: PosterSpec) -> str | None:
    """The poster's real background colour for deriving readable text colours: the darkest gradient stop (worst case)."""
    from poster_agent.checks.pixels import hex_rgb, rel_luminance
    bg = spec.background
    colors = [s.color for s in bg.gradient] if bg.kind == "gradient" and bg.gradient else ([bg.color] if bg.kind == "solid" else [])
    colors = [c for c in colors if isinstance(c, str) and c.startswith("#") and len(c) == 7]
    return min(colors, key=lambda c: rel_luminance(hex_rgb(c))) if colors else None


def solve_layout(spec: PosterSpec, brand: BrandProfile, measurer: TextMeasurer | None = None) -> LayoutResult:
    canvas = spec.canvas
    ctx = Ctx(brand, canvas, measurer or TextMeasurer(), dict(spec.assets), align=spec.align, type_style=spec.type_style, ground=_ground(spec))
    safe = canvas.safe_box
    top = [c for c in spec.components if c.anchor == "top"]
    bottom = [c for c in spec.components if c.anchor == "bottom"]
    flow = [c for c in spec.components if c.anchor == "flow"]
    tpl = get_template(spec.template)
    beside = [c for c in bottom if c.content.get("beside")]  # stands on the bottom edge next to a text column: takes no height from it
    stacked = [c for c in bottom if c not in beside]
    pinned: list[ComponentSpec] = []  # a leading HeaderRow / LogoBlock sits at the top margin; only the rest of the content is moved around
    if len(tpl.zones) == 1 and len(flow) > 1 and flow[0].component in PIN_TOP and flow[0].content.get("pin", "top") == "top":
        pinned, flow = flow[:1], flow[1:]

    def band_heights(s: float) -> tuple[float, float]:
        ctx.scale = s
        th = sum(_component(c).measure(c, ctx, canvas.width) for c in top)
        bh = sum(_component(c).measure(c, ctx, canvas.width) for c in stacked)
        return th, bh

    region = spec.flow_region if tuple(spec.flow_region) != (0.0, 0.0, 1.0, 1.0) or tpl.region is None else tpl.region
    rx0, ry0, rx1, ry1 = region  # flow content stays inside this part of the safe box (bands stay full-bleed)
    fsafe = Box(x=safe.x + rx0 * safe.w, y=safe.y + ry0 * safe.h, w=max(rx1 - rx0, 0.2) * safe.w, h=max(ry1 - ry0, 0.2) * safe.h)

    def pin_box() -> tuple[float, float]:
        """(top, height) of the pinned strip at the current scale."""
        th = sum(_component(c).measure(c, ctx, canvas.width) for c in top)
        return max(safe.y, th + _gap_edge(ctx) if top else safe.y), sum(_component(c).measure(c, ctx, safe.w) for c in pinned)

    def span(s: float) -> tuple[float, float]:
        """Free vertical span between the top and bottom bands at scale `s`."""
        ctx.scale = s
        th, bh = band_heights(s)
        y0 = max(fsafe.y, th + _gap_edge(ctx) if top else fsafe.y)
        y1 = min(fsafe.bottom, canvas.height - bh - _gap_edge(ctx) if bottom else fsafe.bottom)
        if pinned:
            pin_y, pin_h = pin_box()
            y0 = max(y0, pin_y + pin_h + _gap(ctx, flow[0]))
        return y0, y1 - y0

    def groups(y0: float, avail: float) -> list[tuple[list[ComponentSpec], Box, str]]:
        """(components, box, valign) per zone of the template that has components. A full-width zone whose content needs more than its
        share grows, and the zones that start below it give up that height (their bottom edges stay where they were)."""
        out, growth = [], []  # growth: (start fraction of the grown zone, extra height)
        used = {name: [c for c in flow if (c.zone if c.zone in tpl.zones else tpl.default) == name] for name in tpl.zones}
        for name, z in sorted(tpl.zones.items(), key=lambda kv: kv[1].y):
            members = used[name]
            if not members:
                continue
            donors = any(used[o] and oz.y > z.y for o, oz in tpl.zones.items())  # only zones below can give up height
            below = sum(g for start, g in growth if z.y > start)
            y, h = y0 + z.y * avail + below, max(z.h * avail - below, 0.25 * z.h * avail)
            w = z.w * fsafe.w
            if donors and z.w >= 1.0 and group_need(members, w) > h:
                growth.append((z.y, group_need(members, w) - h))
                h = group_need(members, w)
            out.append((members, Box(x=fsafe.x + z.x * fsafe.w, y=y, w=w, h=h), spec.flow_valign if z.valign == "flow" else z.valign))
        return out

    def group_need(members: list[ComponentSpec], w: float) -> float:
        return sum(_component(c).measure(c, ctx, w) for c in members) + sum(_gap(ctx, members[i + 1], tpl.gap_scale)
                                                                          for i in range(len(members) - 1))

    chosen, need, avail, fits = SCALES[-1], 0.0, 0.0, False
    for s in SCALES:
        y0, av = span(s)
        gs = groups(y0, av)
        chosen = s
        worst = max(gs, key=lambda g: group_need(g[0], g[1].w) / max(g[1].h, 1.0), default=None)
        need, avail = (group_need(worst[0], worst[1].w), worst[1].h) if worst else (0.0, av)
        if all(group_need(m, bx.w) <= bx.h for m, bx, _ in gs):
            fits = True
            break

    if fits and chosen == 1.0 and flow and len(groups(*span(1.0))) == 1 and (avail - need) / max(avail, 1.0) > MIN_SLACK_FOR_GROWTH:
        # Fill by scaling: take the largest scale at which everything still fits with a little air (single-zone layouts only).
        for s in UP_SCALES:
            u0, uav = span(s)
            g = groups(u0, uav)[0]
            if group_need(g[0], g[1].w) <= g[1].h * 0.96:
                chosen = s
            else:
                break

    ctx.scale = chosen
    ctx.issues = []
    y0, av = span(chosen)
    th, bh = band_heights(chosen)

    placed: list[PlacedElement] = []
    bleed: set[str] = set()
    cur = 0.0
    for c in top:  # bands pinned to the top edge
        comp = _component(c)
        h = comp.measure(c, ctx, canvas.width)
        els = comp.place(c, ctx, Box(x=0, y=cur, w=canvas.width, h=h))
        placed += els
        bleed |= {e.id for e in els}
        cur += h
    flow_box = None
    elastic_bands = [c for c in stacked if getattr(_component(c), "band_elastic", False)]
    pinned_ids: set[str] = set()
    if pinned:
        py, ph = pin_box()
        for c in pinned:
            els = _component(c).place(c, ctx, Box(x=safe.x, y=py, w=safe.w, h=ph))
            if c.content.get("band"):  # a full-bleed band behind the header: the logo's own field merges into it, the art fades below it
                bar = PlacedElement(id=f"{c.id}.band", kind=ElementKind.shape, role="info_bar", z=1,
                                    box=Box(x=0, y=0, w=canvas.width, h=py + ph + 0.012 * canvas.height), props={"fill": ctx.color(c.content["band"])})
                els.append(bar)
                bleed.add(bar.id)
            placed += els
            pinned_ids |= {e.id for e in els}
    for members, zb, valign in groups(y0, av):
        heights = [_component(c).measure(c, ctx, zb.w) for c in members]
        base_gaps = [_gap(ctx, members[i + 1], tpl.gap_scale) for i in range(len(members) - 1)]
        extra = zb.h - (sum(heights) + sum(base_gaps))
        if extra > 0:
            elastic = [c for c in members if _component(c).elastic and c.content.get("fill", True)]
            if elastic:  # spare height goes to the hero first (keeping ~15% as air), not into wider gaps
                for c in elastic:
                    ctx.extra[c.id] = 0.85 * extra / len(elastic)
                heights = [_component(c).measure(c, ctx, zb.w) for c in members]
                extra = zb.h - (sum(heights) + sum(base_gaps))
        growth = 1.0
        if extra > 0 and base_gaps and not elastic_bands:  # with a bottom anchor the spare height is the anchor's, not the gaps
            growth = min(1.0 + extra / sum(base_gaps), MAX_GAP_GROWTH)
            extra -= sum(base_gaps) * (growth - 1.0)
        y = zb.y + (max(extra, 0.0) * {"top": 0.0, "balanced": 0.35, "center": 0.5, "bottom": 1.0}[valign])  # leftover space
        top_y = y
        for i, c in enumerate(members):
            comp = _component(c)
            placed += comp.place(c, ctx, Box(x=zb.x, y=y, w=zb.w, h=heights[i]))
            y += heights[i] + (base_gaps[i] * growth if i < len(base_gaps) else 0.0)
        gb = Box(x=zb.x, y=top_y, w=zb.w, h=y - top_y)
        flow_box = gb if flow_box is None else Box(x=min(flow_box.x, gb.x), y=min(flow_box.y, gb.y),
                                                   w=max(flow_box.x + flow_box.w, gb.x + gb.w) - min(flow_box.x, gb.x),
                                                   h=max(flow_box.y + flow_box.h, gb.y + gb.h) - min(flow_box.y, gb.y))
        total = sum(heights) + sum(base_gaps)
        if not fits and total > zb.h:
            hs = dict(zip([c.id for c in members], heights))
            tallest = ", ".join(f"{c.id} ({hs[c.id]:.0f}pt)" for c in sorted(members, key=lambda c: -hs[c.id])[:3])
            ctx.issue(IssueCategory.fit,
                      f"content needs {total:.0f}pt but only {zb.h:.0f}pt available at minimum scale {chosen:.2f} "
                      f"({(total / max(zb.h, 1.0) - 1) * 100:.0f}% too tall; tallest components: {tallest})", [c.id for c in members],
                      "everything from the brief must stay, so make components more compact instead of removing text: "
                      "put date/time/venue in one InfoBar variant 'columns', merge small TextBlocks, lower HeroImage "
                      "height_pct, drop optional art (HeroImage/Panel padding), move components to a roomier zone, "
                      "use template classic, or pick a taller format")

    # bottom bands: an elastic anchor (a cutout or photo on the bottom edge) grows up to the content, so the poster has no empty band
    content_bottom = max((e.box.bottom for e in placed if e.id not in bleed and not e.props.get("decorative")), default=y0)
    spare = (canvas.height - bh) - (content_bottom + _gap_edge(ctx))
    if spare > 0 and elastic_bands:
        for c in elastic_bands:
            ctx.extra[c.id] = spare / len(elastic_bands)
    heights = [_component(c).measure(c, ctx, canvas.width) for c in stacked]
    left = spare - (sum(heights) - bh)  # the anchor reached its maximum: centre the content in what is still free, not one big gap
    if left > 0 and elastic_bands:
        moved = [e for e in placed if e.id not in bleed and e.id not in pinned_ids and not e.props.get("decorative")]
        for e in moved:
            e.box = Box(x=e.box.x, y=e.box.y + left / 2, w=e.box.w, h=e.box.h)
        if flow_box is not None:
            flow_box = Box(x=flow_box.x, y=flow_box.y + left / 2, w=flow_box.w, h=flow_box.h)
    cur = canvas.height - sum(heights)
    for c, h in zip(stacked, heights):
        els = _component(c).place(c, ctx, Box(x=0, y=cur, w=canvas.width, h=h))
        placed += els
        bleed |= {e.id for e in els}
        cur += h
    stand = canvas.height - sum(heights)  # a beside-anchor stands on the stacked bands (or the edge)
    for c in beside:
        h = _component(c).measure(c, ctx, canvas.width)
        els = _component(c).place(c, ctx, Box(x=0, y=stand - h, w=canvas.width, h=h))
        placed += els
        bleed |= {e.id for e in els}

    clear = brand.logo_rules.clear_space_mm * MM_TO_PT + 4.0  # brand rule: the logo sits on a flat field with clear space
    keepout = [e.box.expand(clear) for e in placed if e.locked and e.kind == ElementKind.image]
    placed += expand_all(spec.decorations, canvas, brand, ctx.measurer, keepout)  # behind everything (negative z)
    issues = _dedupe(ctx.issues) + planned_geometry_issues(placed, canvas.safe_box, bleed)
    return LayoutResult(placed=placed, assets=ctx.assets, issues=issues, scale=chosen, fits=fits, flow_box=flow_box, bleed_ids=bleed)


def _gap_edge(ctx: Ctx) -> float:
    return BASE_GAP * ctx.canvas.height * ctx.scale


def _dedupe(issues: list[Issue]) -> list[Issue]:
    seen, out = set(), []
    for i in issues:
        key = (i.category, i.evidence)
        if key not in seen:
            seen.add(key)
            out.append(i)
    return out


def planned_geometry_issues(placed: list[PlacedElement], safe: Box, bleed_ids: set[str]) -> list[Issue]:
    """Cheap solver-side sanity check on *planned* boxes. The authoritative check runs on rendered output (M6)."""
    issues: list[Issue] = []
    tol = 0.5
    for el in placed:
        if el.props.get("decorative") or el.id in bleed_ids or el.kind == ElementKind.shape and el.role in {"panel", "photo_tint", "info_bar", "highlight_box"}:
            continue
        if not safe.contains(el.box, tol=tol):
            issues.append(Issue(category=IssueCategory.margin_violation, element_ids=[el.id], source="solver",
                                evidence=f"{el.id} box {el.box.x:.0f},{el.box.y:.0f},{el.box.w:.0f}x{el.box.h:.0f} leaves the safe area"))
    texts = [e for e in placed if e.kind == ElementKind.text and not e.props.get("decorative")]
    for i, a in enumerate(texts):
        for b in texts[i + 1:]:
            area = a.box.intersection_area(b.box)
            if area > 0.04 * min(a.box.w * a.box.h, b.box.w * b.box.h):
                issues.append(Issue(category=IssueCategory.text_overlap, element_ids=[a.id, b.id], source="solver",
                                    evidence=f"planned text boxes {a.id} and {b.id} overlap by {area:.0f}pt²"))
    return issues


def apply_layout(spec: PosterSpec, brand: BrandProfile, measurer: TextMeasurer | None = None) -> LayoutResult:
    """Solve and write the result into the spec (Layout owns `placed`, `assets` additions, and solver issues)."""
    res = solve_layout(spec, brand, measurer)
    for el in res.placed:  # repairs recorded on the spec survive re-layout
        ov = spec.overrides.get(el.id)
        if ov and ov.color:
            el.color = ov.color
    spec.placed = res.placed
    spec.assets = res.assets
    spec.issues = [i for i in spec.issues if i.source != "solver"] + res.issues
    return res
