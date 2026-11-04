"""Hierarchy and proportion checks measured on the placed poster (no pixels, no model): does the event title lead, is anything oversized,
is any text repeated. They answer "does it look designed" with numbers the planner can act on, and still work when no vision model is up.

Measured, never guessed: every issue states the sizes it compared. Findings are typed `hierarchy` / `density` so the router sends them to
the planner. Only posters made from components are examined (a bare list of placed elements has no title to find).
"""
from __future__ import annotations

from poster_agent.spec import ElementKind, Issue, IssueCategory, PlacedElement, PosterSpec, normalize

TITLE_LEAD_MIN = 1.4  # the title's type size over the next-largest other text (the skill asks for 1.6; below 1.4 it is a warning)
TITLE_LEAD_FAIL = 1.18  # ...and below this the title does not lead at all: an error
LOGO_AREA_WARN = 0.06  # share of the poster area
LOGO_AREA_FAIL = 0.09
MIN_REPEAT_CHARS = 3


def _texts(spec: PosterSpec) -> list[PlacedElement]:
    return [e for e in spec.placed if e.kind == ElementKind.text and not e.props.get("decorative") and (e.text or "").strip()]


def title_elements(spec: PosterSpec, texts: list[PlacedElement]) -> list[PlacedElement]:
    """The placed elements that carry the event title (it may be split over several lines): text wholly inside the brief's title."""
    item = next((m for m in spec.mandatory if m.id == "event_title" and m.text), None)
    if item is None:
        return []
    full = normalize(item.text)
    return [e for e in texts if len(normalize(e.text)) >= MIN_REPEAT_CHARS and normalize(e.text) in full]


def _issue(cat: IssueCategory, severity: str, ids: list[str], evidence: str, fix: str) -> Issue:
    return Issue(category=cat, severity=severity, element_ids=ids, evidence=evidence, suggested_fix=fix, source="check")


def title_dominance(spec: PosterSpec) -> tuple[dict, list[Issue]]:
    texts = _texts(spec)
    title = title_elements(spec, texts)
    if not title:
        return {"title_found": False}, []
    t_ids = {e.id for e in title}
    t_size = max(e.size or 0 for e in title)
    others = [e for e in texts if e.id not in t_ids]
    if not others or t_size <= 0:
        return {"title_found": True, "title_size": t_size}, []
    big = max(others, key=lambda e: e.size or 0)
    ratio = t_size / max(big.size or 1, 1e-6)
    measured = {"title_found": True, "title_size": round(t_size, 1), "next_largest": big.id, "next_size": round(big.size or 0, 1),
                "lead_ratio": round(ratio, 2)}
    if ratio >= TITLE_LEAD_MIN:
        return measured, []
    severity = "error" if ratio < TITLE_LEAD_FAIL else "warning"
    return measured, [_issue(
        IssueCategory.hierarchy, severity, [big.id, *sorted(t_ids)],
        f"the event title is set at {t_size:.0f}pt but {big.id} is {big.size:.0f}pt: the title leads by only {ratio:.2f}x "
        f"(needs {TITLE_LEAD_MIN}x)",
        f"make the title clearly the biggest text (a BigTitle) and give {big.id} a smaller role such as subtitle or body")]


def logo_proportion(spec: PosterSpec) -> tuple[dict, list[Issue]]:
    logos = [e for e in spec.placed if e.role == "logo" and e.kind == ElementKind.image]
    if not logos:
        return {}, []
    area = spec.canvas.width * spec.canvas.height
    worst = max(logos, key=lambda e: e.box.w * e.box.h)
    share = worst.box.w * worst.box.h / area
    measured = {"logo_area_share": round(share, 3), "logo_width_share": round(worst.box.w / spec.canvas.width, 3)}
    if share <= LOGO_AREA_WARN:
        return measured, []
    return measured, [_issue(
        IssueCategory.hierarchy, "error" if share > LOGO_AREA_FAIL else "warning", [worst.id],
        f"the logo covers {share:.1%} of the poster ({worst.box.w / spec.canvas.width:.0%} of the width): it competes with the title",
        "reduce the logo width_pct; brand logos stay small")]


def repeated_text(spec: PosterSpec) -> tuple[dict, list[Issue]]:
    """Text that is printed twice: an element whose whole content already appears inside another element (a pill saying '2027' under
    a title ending in '2027')."""
    texts = _texts(spec)
    norm = {e.id: normalize(e.text) for e in texts}
    issues, seen = [], set()
    for a in texts:
        na = norm[a.id]
        if len(na) < MIN_REPEAT_CHARS:
            continue
        for b in texts:
            if a.id != b.id and len(norm[b.id]) > len(na) and na in norm[b.id] and (a.id, b.id) not in seen:
                seen.add((a.id, b.id))
                issues.append(_issue(IssueCategory.density, "warning", [a.id, b.id],
                                     f"{a.id} repeats text already printed in {b.id} ({(a.text or '').strip()[:30]!r})",
                                     f"remove {a.id} or use a different brief line there"))
                break
    return {"repeated": len(issues)}, issues


CONTAINER_MIN_SHARE = 0.06  # a filled container this big (share of the poster area) must actually hold content
CONTAINER_MIN_FILL = 0.22  # ...with at least this share of the container covered by its content
CONTAINER_ROLES = {"panel", "highlight_box", "image_card"}


def container_fill(spec: PosterSpec) -> tuple[dict, list[Issue]]:
    """Big filled panels whose content covers only a sliver of them (a small logo in a huge white box)."""
    area = spec.canvas.width * spec.canvas.height
    issues, worst = [], 1.0
    for s in spec.placed:
        fill = s.props.get("fill")
        if s.kind != ElementKind.shape or s.role not in CONTAINER_ROLES or not fill or s.box.w * s.box.h < CONTAINER_MIN_SHARE * area:
            continue
        inside = [e for e in spec.placed if e.kind != ElementKind.shape and e.z > s.z and s.box.x <= e.box.cx <= s.box.right
                  and s.box.y <= e.box.cy <= s.box.bottom]
        covered = sum(e.box.intersection_area(s.box) for e in inside) / max(s.box.w * s.box.h, 1.0)
        worst = min(worst, covered)
        if covered < CONTAINER_MIN_FILL:
            issues.append(_issue(IssueCategory.density, "error", [s.id, *[e.id for e in inside][:3]],
                                 f"{s.id} covers {s.box.w * s.box.h / area:.0%} of the poster but its content fills only {covered:.0%} of it",
                                 "remove the container or put more of the content in it; never wrap a lone logo in a panel"))
    return {"worst_fill": round(worst, 3), "empty_containers": len(issues)}, issues


# components that legitimately set their own alignment inside a poster (cells, bands, header lines, two-column blocks)
FREE_ALIGN = {"IconRow", "InfoBar", "TabBanner", "HighlightBox", "Columns", "HeaderRow", "BigWord", "PartnerLogoRow", "PhotoStrip",
              "SpeakerGrid", "ChiefGuestCard", "ImageBlock", "PhotoFade"}
AXIS_TOL = 0.004  # of the canvas width: edges closer than this are the same axis
MAX_LEFT_AXES = 3


def _walk(components):
    for c in components:
        yield c
        yield from _walk(c.children)


def _cluster(values: list[float], tol: float) -> list[float]:
    out: list[float] = []
    for v in sorted(values):
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


def alignment(spec: PosterSpec) -> tuple[dict, list[Issue]]:
    """One alignment axis per poster: text blocks follow the poster's own alignment (left or centred), and left edges fall on few axes."""
    kinds, free = {}, set()
    for c in _walk(spec.components):
        kinds[c.id] = c.component
        if c.component in FREE_ALIGN:
            free |= {d.id for d in _walk([c])}
    texts = [e for e in _texts(spec) if (e.source_component or "") not in free]
    odd = [e for e in texts if e.align != spec.align and e.role != "adapt_scrim"]
    tol = AXIS_TOL * spec.canvas.width
    lefts = [e.box.x for e in texts if e.align == "left"] + [e.box.x for e in spec.placed if e.role == "logo" and e.box.cx < spec.canvas.width / 2]
    axes = _cluster(lefts, tol)
    measured = {"poster_align": spec.align, "off_axis_text": [e.id for e in odd], "left_axes": [round(a, 1) for a in axes]}
    issues: list[Issue] = []
    if odd:
        issues.append(_issue(IssueCategory.spacing, "error" if len(odd) >= 3 else "warning", [e.id for e in odd[:6]],
                             f"mixed alignment: {', '.join(e.id for e in odd[:4])} are {odd[0].align}-aligned in a {spec.align}-aligned "
                             "poster (one alignment axis per poster)", f"align them {spec.align}, or move them into a Columns / HeaderRow block"))
    if len(axes) > MAX_LEFT_AXES:
        issues.append(_issue(IssueCategory.spacing, "warning", [], f"{len(axes)} different left edges ({', '.join(f'{a:.0f}' for a in axes)}pt): "
                             f"text and logo should share one or two", "align the blocks to the poster margin"))
    return measured, issues


def measure(spec: PosterSpec) -> list[tuple[str, bool, str, dict, list[Issue]]]:
    """(check name, passed, detail, measured, issues) for each aesthetics check; empty for specs without components."""
    if not spec.components:
        return []
    out = []
    m, iss = title_dominance(spec)
    if m.get("title_found") and "lead_ratio" in m:
        out.append(("title_dominance", not any(i.severity == "error" for i in iss),
                    f"title {m['title_size']}pt vs next largest {m['next_largest']} {m['next_size']}pt: leads {m['lead_ratio']}x "
                    f"(needs {TITLE_LEAD_MIN}x)", m, iss))
    m, iss = logo_proportion(spec)
    if m:
        out.append(("logo_proportion", not any(i.severity == "error" for i in iss),
                    f"logo covers {m['logo_area_share']:.1%} of the poster ({m['logo_width_share']:.0%} of the width)", m, iss))
    m, iss = repeated_text(spec)
    out.append(("repeated_text", not iss, f"{m['repeated']} repeated text element(s)", m, iss))
    m, iss = alignment(spec)
    out.append(("alignment", not any(i.severity == "error" for i in iss),
                f"{len(m['left_axes'])} left axis/axes {m['left_axes']}; {len(m['off_axis_text'])} text element(s) off the poster's "
                f"{m['poster_align']} alignment", m, iss))
    m, iss = container_fill(spec)
    out.append(("container_fill", not iss, f"{m['empty_containers']} oversized container(s) with little content "
                f"(emptiest holds {m['worst_fill']:.0%} of its area)", m, iss))
    return out
