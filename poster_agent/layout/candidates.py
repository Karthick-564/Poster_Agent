"""Choose where the text goes AFTER the art exists.

The planner picks components and content; this stage tries many placements of that same content against what is actually in the
background (`analysis.ArtMap`) and keeps the best: a left or right column or the full width, the upper, middle or lower part of the
poster, content packed to the top / middle / bottom of that window. Each candidate is solved by the normal deterministic solver (so it
is a legal layout, or it is dropped) and scored from measurements of the art under the text. The planner's own placement is always a
candidate and is kept unless another clearly beats it, so a poster whose art is already fine is not shuffled for nothing."""
from __future__ import annotations

from dataclasses import dataclass, field

from poster_agent.analysis import ArtMap
from poster_agent.spec import ElementKind, IssueCategory, PlacedElement, PosterSpec
from poster_agent.spec.brand import BrandProfile

from .fonts import TextMeasurer
from .solver import solve_layout

COLUMNS = [(0.0, 1.0), (0.0, 0.66), (0.34, 1.0)]  # x range of the flow region
WINDOWS = [(0.0, 1.0), (0.0, 0.62), (0.19, 0.81), (0.38, 1.0)]  # y range
VALIGNS = ["top", "balanced", "center", "bottom"]
BLOCKING = {IssueCategory.fit, IssueCategory.margin_violation, IssueCategory.text_overlap}
MIN_GAIN = 0.05  # a candidate must beat the planner's placement by this much (absolute) and by 10% to replace it


@dataclass
class Candidate:
    region: tuple[float, float, float, float]
    valign: str
    cost: float
    terms: dict[str, float]


@dataclass
class LayoutChoice:
    chosen: Candidate
    baseline: Candidate | None
    tried: int
    legal: int
    ranked: list[Candidate] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.baseline is not None and (self.chosen.region, self.chosen.valign) != (self.baseline.region, self.baseline.valign)

    def note(self) -> str:
        c, b = self.chosen, self.baseline
        where = f"x {c.region[0]:.0%}-{c.region[2]:.0%}, y {c.region[1]:.0%}-{c.region[3]:.0%}, {c.valign}"
        if not self.changed:
            return (f"layout kept as planned (placement cost {c.cost:.2f}; {self.legal} of {self.tried} candidate placements were legal "
                    f"and none beat it clearly)")
        return (f"layout moved to fit the art: {where} (placement cost {c.cost:.2f} against {b.cost:.2f} as planned; "
                f"busyness {c.terms['busy']:.2f} vs {b.terms['busy']:.2f} and subject overlap {c.terms['subject']:.2f} vs "
                f"{b.terms['subject']:.2f} under the text; {self.legal} of {self.tried} candidate placements were legal)")


def _edge_bands(placed: list[PlacedElement], height: float) -> list[tuple[float, float]]:
    """(y0, y1) of the zones where art fades into a full-width opaque band at the top or bottom edge (see layout/adapt.py)."""
    from .adapt import FADE_FRACTION
    out = []
    for e in placed:
        if e.kind == ElementKind.shape and e.role == "info_bar" and e.box.w >= 0.99 * max(x.box.right for x in placed):
            if e.box.bottom >= height - 1:
                out.append((e.box.y - FADE_FRACTION * height, e.box.y))
            elif e.box.y <= 1:
                out.append((e.box.bottom, e.box.bottom + FADE_FRACTION * height))
    return out


def score(placed: list[PlacedElement], art: ArtMap) -> tuple[float, dict[str, float]]:
    """Lower is better. Measured from the art under the boxes and from the geometry: busy texture under the text, overlap with the
    subject, text on the subject's side of the poster, a block squeezed into too little height, the logo floating in the middle of
    the poster instead of belonging to an edge, and text sitting in the zone where the art fades into an edge band."""
    texts = [e for e in placed if e.kind == ElementKind.text and not e.props.get("decorative")]
    if not texts:
        return 0.0, {"busy": 0.0, "subject": 0.0, "bias": 0.0, "spread": 0.0, "logo_float": 0.0, "fade": 0.0}
    area = sum(e.box.w * e.box.h for e in texts) or 1.0
    busy = sum(e.box.w * e.box.h * art.busy_in(e.box) for e in texts) / area
    subj = sum(e.box.w * e.box.h * art.subject_in(e.box) for e in texts) / area
    tx = sum(e.box.cx * e.box.w * e.box.h for e in texts) / area
    spread = (max(e.box.bottom for e in texts) - min(e.box.y for e in texts)) / art.height
    cen = art.subject_centroid()
    bias = max(0.0, 0.25 - abs(tx - cen[0]) / art.width) if cen else 0.0
    logos = [e for e in placed if e.role == "logo" and e.kind == ElementKind.image]
    logo_float = max((1.0 - abs(e.box.cy / art.height - 0.5) / 0.25 for e in logos), default=0.0)  # 1 at mid-height, 0 beyond the outer quarters
    logo_float = min(max(logo_float, 0.0), 1.0)
    fades = _edge_bands(placed, art.height)
    in_fade = sum(e.box.w * max(0.0, min(e.box.bottom, y1) - max(e.box.y, y0)) for e in texts for y0, y1 in fades
                  ) / area
    terms = {"busy": busy, "subject": subj, "bias": bias, "spread": max(0.0, 0.5 - spread), "logo_float": logo_float,
             "fade": min(in_fade, 1.0)}
    cost = busy + 2.0 * subj + 1.0 * bias + 0.3 * terms["spread"] + 0.6 * logo_float + 1.2 * terms["fade"]
    return cost, terms


def _try(spec: PosterSpec, brand: BrandProfile, art: ArtMap, region, valign, measurer) -> Candidate | None:
    sp = spec.model_copy(update={"flow_region": region, "flow_valign": valign})
    res = solve_layout(sp, brand, measurer)
    if not res.fits or any(i.category in BLOCKING for i in res.issues):
        return None
    cost, terms = score(res.placed, art)
    return Candidate(region, valign, cost, terms)


STEP = 0.04  # how far a region edge moves in one refinement step (fraction of the safe box)
MIN_SPAN = 0.4  # a region never narrows or shortens below this fraction


def refine(spec: PosterSpec, brand: BrandProfile, art: ArtMap, start: Candidate, measurer, passes: int = 6) -> Candidate:
    """Continuous local search from `start`: nudge each edge of the region (and try the other vertical alignments), keep any legal change
    that lowers the cost, stop when nothing improves. This is what lets the placement follow THIS picture instead of a fixed set of
    positions: the text block's width and window settle where the art under it is calmest."""
    cur = start
    for _ in range(passes):
        x0, y0, x1, y1 = cur.region
        moves = []
        for i, d in [(0, STEP), (0, -STEP), (1, STEP), (1, -STEP), (2, STEP), (2, -STEP), (3, STEP), (3, -STEP)]:
            r = [x0, y0, x1, y1]
            r[i] = min(max(r[i] + d, 0.0), 1.0)
            if r[2] - r[0] >= MIN_SPAN and r[3] - r[1] >= MIN_SPAN:
                moves.append((tuple(round(v, 3) for v in r), cur.valign))
        moves += [(cur.region, v) for v in VALIGNS if v != cur.valign]
        best = cur
        for region, v in moves:
            c = _try(spec, brand, art, region, v, measurer)
            if c is not None and c.cost < best.cost - 0.005:
                best = c
        if best is cur:
            break
        cur = best
    return cur


def choose_layout(spec: PosterSpec, brand: BrandProfile, art: ArtMap, measurer: TextMeasurer | None = None) -> LayoutChoice | None:
    """Best legal placement of the spec's flow content against `art`; None when there is no flow content to place."""
    if not any(c.anchor == "flow" for c in spec.components):
        return None
    measurer = measurer or TextMeasurer()
    base = _try(spec, brand, art, spec.flow_region, spec.flow_valign, measurer)
    cands = [base] if base else []
    tried = 1
    for x0, x1 in COLUMNS:
        for y0, y1 in WINDOWS:
            for v in VALIGNS:
                region = (x0, y0, x1, y1)
                if region == tuple(spec.flow_region) and v == spec.flow_valign:
                    continue
                tried += 1
                c = _try(spec, brand, art, region, v, measurer)
                if c:
                    cands.append(c)
    if not cands:
        return None
    ranked = sorted(cands, key=lambda c: c.cost)
    refined = [refine(spec, brand, art, c, measurer) for c in ranked[:3]]  # the three best starts, each polished continuously
    ranked = sorted(ranked + refined, key=lambda c: c.cost)
    best = ranked[0]
    if base is not None and not (best.cost < base.cost * 0.9 - MIN_GAIN):
        best = base
    return LayoutChoice(chosen=best, baseline=base, tried=tried, legal=len(cands), ranked=ranked[:5])


def apply_choice(spec: PosterSpec, choice: LayoutChoice) -> None:
    spec.flow_region = choice.chosen.region
    spec.flow_valign = choice.chosen.valign
