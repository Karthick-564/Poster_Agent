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

from . import components, structure  # noqa: F401  (registers components)
from .decor import expand_all
from .base import REGISTRY, Ctx
from .fonts import TextMeasurer

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
    ctx = Ctx(brand, canvas, measurer or TextMeasurer(), dict(spec.assets), align=spec.align, ground=_ground(spec))
    safe = canvas.safe_box
    top = [c for c in spec.components if c.anchor == "top"]
    bottom = [c for c in spec.components if c.anchor == "bottom"]
    flow = [c for c in spec.components if c.anchor == "flow"]

    def band_heights(s: float) -> tuple[float, float]:
        ctx.scale = s
        th = sum(_component(c).measure(c, ctx, canvas.width) for c in top)
        bh = sum(_component(c).measure(c, ctx, canvas.width) for c in bottom)
        return th, bh

    chosen, need, avail, fits = SCALES[-1], 0.0, 0.0, False
    for s in SCALES:
        ctx.scale = s
        th, bh = band_heights(s)
        y0 = max(safe.y, th + _gap_edge(ctx) if top else safe.y)
        y1 = min(safe.bottom, canvas.height - bh - _gap_edge(ctx) if bottom else safe.bottom)
        heights = [_component(c).measure(c, ctx, safe.w) for c in flow]
        gaps = [_gap(ctx, flow[i + 1]) for i in range(len(flow) - 1)]
        need, avail = sum(heights) + sum(gaps), y1 - y0
        chosen = s
        if need <= avail:
            fits = True
            break

    if fits and chosen == 1.0 and flow and (avail - need) / max(avail, 1.0) > MIN_SLACK_FOR_GROWTH:
        # Fill by scaling: take the largest scale at which everything still fits with a little air.
        for s in UP_SCALES:
            ctx.scale = s
            th, bh = band_heights(s)
            u0 = max(safe.y, th + _gap_edge(ctx) if top else safe.y)
            u1 = min(safe.bottom, canvas.height - bh - _gap_edge(ctx) if bottom else safe.bottom)
            need_s = sum(_component(c).measure(c, ctx, safe.w) for c in flow) + sum(
                _gap(ctx, flow[i + 1]) for i in range(len(flow) - 1))
            if need_s <= (u1 - u0) * 0.96:
                chosen = s
            else:
                break

    ctx.scale = chosen
    ctx.issues = []
    th, bh = band_heights(chosen)
    y0 = max(safe.y, th + _gap_edge(ctx) if top else safe.y)
    y1 = min(safe.bottom, canvas.height - bh - _gap_edge(ctx) if bottom else safe.bottom)
    avail = y1 - y0

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
    cur = canvas.height - bh
    for c in bottom:
        comp = _component(c)
        h = comp.measure(c, ctx, canvas.width)
        els = comp.place(c, ctx, Box(x=0, y=cur, w=canvas.width, h=h))
        placed += els
        bleed |= {e.id for e in els}
        cur += h

    flow_box = None
    if flow:
        heights = [_component(c).measure(c, ctx, safe.w) for c in flow]
        base_gaps = [_gap(ctx, flow[i + 1]) for i in range(len(flow) - 1)]
        extra = avail - (sum(heights) + sum(base_gaps))
        if extra > 0:
            elastic = [c for c in flow if _component(c).elastic and c.content.get("fill", True)]
            if elastic:  # spare height goes to the hero first (keeping ~15% as air), not into wider gaps
                for c in elastic:
                    ctx.extra[c.id] = 0.85 * extra / len(elastic)
                heights = [_component(c).measure(c, ctx, safe.w) for c in flow]
                extra = avail - (sum(heights) + sum(base_gaps))
        growth = 1.0
        if extra > 0 and base_gaps:
            growth = min(1.0 + extra / sum(base_gaps), MAX_GAP_GROWTH)
            extra -= sum(base_gaps) * (growth - 1.0)
        y = y0 + (max(extra, 0.0) * 0.35)  # leftover space: a bit more below than above
        top_y = y
        for i, c in enumerate(flow):
            comp = _component(c)
            placed += comp.place(c, ctx, Box(x=safe.x, y=y, w=safe.w, h=heights[i]))
            y += heights[i] + (base_gaps[i] * growth if i < len(base_gaps) else 0.0)
        flow_box = Box(x=safe.x, y=top_y, w=safe.w, h=y - top_y)
        if not fits:
            tall = sorted(zip(flow, heights), key=lambda t: -t[1])[:3]
            over = (need / avail - 1) * 100 if avail > 0 else 100
            tallest = ", ".join(f"{c.id} ({h:.0f}pt)" for c, h in tall)
            ctx.issue(IssueCategory.fit,
                      f"content needs {need:.0f}pt but only {avail:.0f}pt available at minimum scale {chosen:.2f} "
                      f"({over:.0f}% too tall; tallest components: {tallest})", [c.id for c in flow],
                      "everything from the brief must stay, so make components more compact instead of removing text: "
                      "put date/time/venue in one InfoBar variant 'columns', merge small TextBlocks, lower HeroImage "
                      "height_pct, drop optional art (HeroImage/Panel padding), or pick a taller format")

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
        if el.props.get("decorative") or el.id in bleed_ids or el.kind == ElementKind.shape and el.role in {"panel", "photo_tint", "info_bar"}:
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
