"""run_checks: verify the *rendered* poster. Every issue carries the measured evidence (guardrail 6: reports state only what
a check measured). Checks: output spec, VectorCraft overflow/font warnings, geometry on real bounds, per-element OCR,
mandatory content, stray text, contrast under text, brand-logo rules (fidelity, size, clear space, solid field)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from poster_agent.compile.compiler import CompileResult
from poster_agent.compile.vc_client import ROOT
from poster_agent.spec import AssetKind, Box, ElementKind, Issue, IssueCategory, PlacedElement, PosterSpec
from poster_agent.spec.brand import BrandProfile
from poster_agent.spec.formats import MM_TO_PT

from . import emptiness, ocr, pixels

LARGE_TEXT_PT = 24.0  # WCAG "large text" relaxation (3:1)
LARGE_TEXT_RATIO = 3.0
OCR_ELEMENT_FAIL = 60.0  # element text effectively absent / illegible
OCR_ELEMENT_WARN = 80.0
OCR_MANDATORY_MIN = 85.0
LOGO_CORRELATION_MIN = 0.90
SOLID_STD_MAX = 10.0  # luminance std (0-255) around a logo above which the field is not flat
SOLID_SPREAD_MAX = 14.0
EDGE_MARGIN_PCT = 1.5  # text must keep this much clearance (of the short side) from every canvas edge
BAND_ROLES = {"info_bar", "photo", "photo_tint", "panel", "divider", "highlight_box"}
OCCLUDING_ROLES = {"logo", "partner_logo", "hero", "headshot"}
SUPPLIED_KINDS = {AssetKind.logo, AssetKind.partner_logo, AssetKind.headshot, AssetKind.photo}
# dead space: thresholds calibrated on the designers' posters (5-26% empty, median ~15%) and on our own plain renders (25-31%)
EMPTY_FRACTION_MAX = 0.24
LARGEST_REGION_MAX = 0.14
BAND_MAX = 0.035
FILLED_ROLES = {"info_bar", "highlight_box", "photo_tint", "photo", "hero", "headshot", "headshot_ring", "logo", "logo_plate",
                "partner_logo", "hero_frame", "icon"}  # NOT 'panel': a mostly-empty panel is dead space


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    measured: dict = field(default_factory=dict)


@dataclass
class CheckReport:
    issues: list[Issue] = field(default_factory=list)
    results: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "results": [r.__dict__ for r in self.results],
                "issues": [i.model_dump(mode="json") for i in self.issues]}


def _bounds(el: PlacedElement, compiled: CompileResult) -> Box:
    return compiled.bounds.get(el.id, el.box)


def _issue(cat: IssueCategory, evidence: str, ids: list[str] | None = None, fix: str | None = None,
           severity: str = "error") -> Issue:
    return Issue(category=cat, severity=severity, element_ids=ids or [], evidence=evidence, suggested_fix=fix, source="check")


def run_checks(spec: PosterSpec, brand: BrandProfile, compiled: CompileResult, *, root: Path = ROOT,
               use_ocr: bool = True, use_emptiness: bool = True) -> CheckReport:
    rep = CheckReport()
    canvas = spec.canvas
    scale = canvas.export_scale
    img = Image.open(compiled.png).convert("RGB")
    texts = [e for e in spec.placed if e.kind == ElementKind.text and not e.props.get("decorative")]  # decoration is exempt
    by_id = {e.id: e for e in spec.placed}

    # ---- 1. output spec -----------------------------------------------------------------------------------------
    want = (round(canvas.width * scale), round(canvas.height * scale))
    ok = abs(img.size[0] - want[0]) <= 1 and abs(img.size[1] - want[1]) <= 1
    if not ok:
        rep.issues.append(_issue(IssueCategory.output_spec, f"PNG is {img.size[0]}x{img.size[1]}px, expected {want[0]}x{want[1]}px"))
    for path, label in ((compiled.pdf, "PDF"), (compiled.vectorcraft, ".vectorcraft")):
        if path is None or not Path(path).exists():
            ok = False
            rep.issues.append(_issue(IssueCategory.output_spec, f"{label} output is missing"))
    fonts = [w for w in compiled.warnings if "font" in w.lower()]
    if fonts:
        ok = False
        rep.issues.append(_issue(IssueCategory.output_spec, "VectorCraft reports a font fallback: " + fonts[0][:160],
                                 fix="install the brand font"))
    rep.results.append(CheckResult("output_spec", ok and not fonts, f"png {img.size[0]}x{img.size[1]} (want {want[0]}x{want[1]}), "
                                   f"{len(fonts)} font warning(s)", {"size": list(img.size)}))

    # ---- 2. text fit (VectorCraft's own overflow flags) ----------------------------------------------------------
    for eid in compiled.overflow:
        rep.issues.append(_issue(IssueCategory.fit, f"{eid}: text does not fit its frame (VectorCraft overflow)", [eid],
                                 "enlarge the frame, shrink the text or shorten it"))
    rep.results.append(CheckResult("text_fit", not compiled.overflow, f"{len(compiled.overflow)} overflowing text element(s)",
                                   {"overflow": list(compiled.overflow)}))

    # ---- 3. geometry on real bounds ------------------------------------------------------------------------------
    edge = EDGE_MARGIN_PCT / 100.0 * min(canvas.width, canvas.height)
    geo_issues = 0
    for el in texts:
        b = _bounds(el, compiled)
        if b.x < edge - 0.5 or b.y < edge - 0.5 or b.right > canvas.width - edge + 0.5 or b.bottom > canvas.height - edge + 0.5:
            geo_issues += 1
            rep.issues.append(_issue(IssueCategory.margin_violation,
                                     f"{el.id} real bounds {b.x:.0f},{b.y:.0f}..{b.right:.0f},{b.bottom:.0f} are within "
                                     f"{edge:.0f}pt of a canvas edge", [el.id], "move it inward"))
    occluders = [e for e in spec.placed if e.role in OCCLUDING_ROLES]
    for i, a in enumerate(texts):
        ba = _bounds(a, compiled)
        for other in texts[i + 1:]:
            bb = _bounds(other, compiled)
            area = ba.intersection_area(bb)
            if area > 0.04 * min(ba.w * ba.h, bb.w * bb.h):
                geo_issues += 1
                rep.issues.append(_issue(IssueCategory.text_overlap, f"{a.id} and {other.id} overlap by {area:.0f}pt²", [a.id, other.id]))
        for occ in occluders:
            bo = _bounds(occ, compiled)
            area = ba.intersection_area(bo)
            if area > 0.04 * ba.w * ba.h:
                geo_issues += 1
                rep.issues.append(_issue(IssueCategory.text_overlap, f"text {a.id} overlaps {occ.role} {occ.id} by {area:.0f}pt²",
                                         [a.id, occ.id]))
    rep.results.append(CheckResult("geometry", geo_issues == 0, f"{len(texts)} text elements vs margins/overlaps: {geo_issues} issue(s)",
                                   {"edge_margin_pt": round(edge, 1)}))

    # ---- 4. OCR: per-element legibility, mandatory content, stray text ---------------------------------------------
    if use_ocr:
        _ocr_checks(spec, compiled, img, scale, texts, rep)
    else:
        rep.results.append(CheckResult("ocr", True, "skipped (use_ocr=False)"))

    # ---- 5. contrast under text -----------------------------------------------------------------------------------
    worst, n_fail, n_meas = None, 0, 0
    for el in texts:
        if not el.color:
            continue
        m = pixels.text_contrast(img, _bounds(el, compiled), scale, pixels.hex_rgb(el.color))
        if m["ratio"] is None:
            continue
        n_meas += 1
        need = LARGE_TEXT_RATIO if (el.size or 0) >= LARGE_TEXT_PT else brand.min_text_contrast
        if worst is None or m["ratio"] / need < worst[0]:
            worst = (m["ratio"] / need, el.id, m["ratio"], need)
        if m["ratio"] < need:
            n_fail += 1
            rep.issues.append(_issue(
                IssueCategory.low_contrast,
                f"{el.id}: contrast {m['ratio']:.2f}:1 of {el.color} over the rendered background (needs {need:.1f}:1)",
                [el.id], "change the text colour, add a scrim, or move the text to a calmer area"))
    rep.results.append(CheckResult("contrast", n_fail == 0, f"{n_meas} text elements measured, {n_fail} below threshold"
                                   + (f"; tightest {worst[1]} at {worst[2]:.2f}:1 (needs {worst[3]:.1f})" if worst else ""),
                                   {"measured": n_meas, "failed": n_fail}))

    # ---- 5b. dead space ---------------------------------------------------------------------------------------------
    filled = [_bounds(e, compiled) for e in spec.placed if not e.props.get("decorative")
              and (e.kind in (ElementKind.image, ElementKind.qr) or e.role in FILLED_ROLES)]
    emp = emptiness.measure_emptiness(img, scale, filled) if use_emptiness else emptiness.EmptinessReport(0.0, 0.0, 0.0)
    over = []
    if emp.empty_fraction > EMPTY_FRACTION_MAX:
        over.append(f"{emp.empty_fraction:.0%} of the poster is empty (limit {EMPTY_FRACTION_MAX:.0%})")
    if emp.largest_fraction > LARGEST_REGION_MAX:
        over.append(f"one empty region covers {emp.largest_fraction:.0%} of the poster (limit {LARGEST_REGION_MAX:.0%})")
    if emp.longest_band_fraction > BAND_MAX:
        over.append(f"an empty horizontal band is {emp.longest_band_fraction:.0%} of the poster height (limit {BAND_MAX:.1%})")
    regions = [{"x": b.x, "y": b.y, "w": b.w, "h": b.h, "fraction": round(f, 4)} for b, f in emp.regions]
    rep.results.append(CheckResult("emptiness", not over, "; ".join(over) or f"{emp.empty_fraction:.0%} empty, largest region "
                                   f"{emp.largest_fraction:.0%}", {"empty_fraction": round(emp.empty_fraction, 4),
                                   "largest_fraction": round(emp.largest_fraction, 4),
                                   "longest_band_fraction": round(emp.longest_band_fraction, 4), "regions": regions}))
    if over:
        rep.issues.append(_issue(IssueCategory.dead_space, "dead space: " + "; ".join(over), [],
                                 "add brand decoration over the empty regions, enlarge the hero/type, or add a visual element"))

    # ---- 6. logos ---------------------------------------------------------------------------------------------------
    _logo_checks(spec, brand, compiled, img, scale, rep, root, by_id)
    return rep


def _ocr_checks(spec, compiled, img, scale, texts, rep: CheckReport) -> None:
    per: dict[str, str] = {}
    unverifiable, weak, failed = [], 0, 0
    for el in texts:
        if not ocr.verifiable(el.text or ""):
            unverifiable.append(el.id)
            continue
        b = _bounds(el, compiled)
        got = ocr.read_crop(img, pixels.pixel_box(b, scale, img.size, pad=3))
        per[el.id] = got
        score = ocr.match_score(el.text, got)
        if score < OCR_ELEMENT_FAIL:
            failed += 1
            rep.issues.append(_issue(IssueCategory.mandatory_missing,
                                     f"{el.id}: expected {el.text!r} but OCR read {got[:80]!r} (match {score:.0f}/100)", [el.id],
                                     "text is missing, hidden or illegible in the render"))
        elif score < OCR_ELEMENT_WARN:
            weak += 1
            rep.issues.append(_issue(IssueCategory.mandatory_missing,
                                     f"{el.id}: only partly legible, OCR read {got[:80]!r} (match {score:.0f}/100)", [el.id],
                                     "increase size or contrast", severity="warning"))
    rep.results.append(CheckResult("ocr_elements", failed == 0, f"{len(per)} text elements OCR'd: {failed} missing/illegible, "
                                   f"{weak} weak; {len(unverifiable)} unverifiable (non-Latin or too short)",
                                   {"unverifiable": unverifiable}))

    corpus = " ".join(per[e.id] for e in sorted((t for t in texts if t.id in per), key=lambda t: (_bounds(t, compiled).y, _bounds(t, compiled).x)))
    miss, checked = [], 0
    for m in spec.mandatory:
        if m.kind != "text" or not m.text:
            continue
        if not ocr.verifiable(m.text):
            continue
        checked += 1
        s = ocr.match_score(m.text, corpus)
        if s < OCR_MANDATORY_MIN:
            miss.append(m.id)
            rep.issues.append(_issue(IssueCategory.mandatory_missing,
                                     f"mandatory {m.label}: {m.text!r} not found in the rendered poster (best match {s:.0f}/100)",
                                     fix="the text is missing or illegible"))
    rep.results.append(CheckResult("mandatory_text", not miss, f"{checked} mandatory text items checked against the render, "
                                   f"{len(miss)} missing", {"missing": miss}))

    # stray text: OCR the whole render; anything outside the places we put text/logos/supplied photos is unexpected
    lines = ocr.read_lines(img, scale=scale)
    zones: list[Box] = [_bounds(t, compiled).expand(6) for t in texts]
    zones += [_bounds(e, compiled).expand(6) for e in spec.placed if e.kind == ElementKind.text and e.props.get("decorative")]
    for e in spec.placed:
        if e.kind == ElementKind.image and spec.assets[e.asset_id].kind in SUPPLIED_KINDS:
            zones.append(_bounds(e, compiled))
    strays = []
    for ln in lines:
        if ln.conf < 0.6 or not ocr.verifiable(ln.text):
            continue
        cx, cy = ln.cx, ln.cy
        if any(z.x <= cx <= z.right and z.y <= cy <= z.bottom for z in zones):
            continue
        strays.append(ln)
        near = [e.id for e in spec.placed if e.kind == ElementKind.image and _bounds(e, compiled).x <= cx <= _bounds(e, compiled).right
                and _bounds(e, compiled).y <= cy <= _bounds(e, compiled).bottom]
        rep.issues.append(_issue(IssueCategory.stray_text, f"unexpected text {ln.text!r} (conf {ln.conf:.2f}) at "
                                 f"{ln.x0:.0f},{ln.y0:.0f} where none was placed", near or (["background"] if spec.background.asset_id else []),
                                 "regenerate the art that contains it"))
    rep.results.append(CheckResult("stray_text", not strays, f"{len(lines)} OCR lines on the whole render, {len(strays)} unexpected",
                                   {"strays": [s.text for s in strays]}))


def _logo_checks(spec, brand, compiled, img, scale, rep: CheckReport, root: Path, by_id) -> None:
    canvas = spec.canvas
    clear = brand.logo_rules.clear_space_mm * MM_TO_PT
    logos = [e for e in spec.placed if e.kind == ElementKind.image and e.role in ("logo", "partner_logo")]
    n_bad = 0
    for el in logos:
        b = _bounds(el, compiled)
        asset = spec.assets[el.asset_id]
        problems: list[str] = []
        try:
            art = Image.open(root / asset.path)
            corr = pixels.logo_correlation(img, b, scale, art)
            if corr < LOGO_CORRELATION_MIN:
                problems.append(f"rendered logo does not match the artwork (correlation {corr:.2f}; covered, distorted or recoloured)")
        except OSError:
            corr = None
            problems.append("logo artwork file is unreadable")

        # clear space: nothing else within the exclusion zone (containers/rules exempt)
        zone = b.expand(clear)
        intruders = []
        for other in spec.placed:
            if other.id == el.id or other.role in ("divider", "photo_tint") or other.props.get("decorative"):
                continue
            ob = _bounds(other, compiled)
            if ob.contains(zone, tol=0.5) or not zone.intersects(ob):
                continue
            intruders.append(other.id)
        if intruders:
            problems.append(f"{', '.join(intruders[:4])} inside the {brand.logo_rules.clear_space_mm:g} mm exclusion zone")
        if b.x < clear - 0.5 or b.y < clear - 0.5 or b.right > canvas.width - clear + 0.5 or b.bottom > canvas.height - clear + 0.5:
            problems.append(f"closer than {brand.logo_rules.clear_space_mm:g} mm to the canvas edge")

        if asset.id.startswith("brand_logo_"):
            var = brand.logo(asset.id[len("brand_logo_"):])
            pct = b.w / canvas.width * 100
            if pct < var.min_width_pct - 0.1:
                problems.append(f"logo is {pct:.1f}% of the canvas width, below the {var.min_width_pct:g}% minimum")
            if abs((b.w / b.h) / var.aspect - 1) > 0.02:
                problems.append(f"logo aspect ratio {b.w / b.h:.2f} differs from the artwork's {var.aspect:.2f}")
            if not brand.logo_rules.allow_gradient_behind or var.field != "any":
                strips = pixels.ring_strips(b, clear * 0.8, canvas.width, canvas.height)
                solid = pixels.field_solidity(img, strips, scale)
                if solid["max_std"] > SOLID_STD_MAX or solid["mean_spread"] > SOLID_SPREAD_MAX:
                    problems.append(f"logo is not on a flat field (luminance std {solid['max_std']:.1f}, spread {solid['mean_spread']:.1f})")
                if var.field == "light" and solid["mean_lum"] < 140:
                    problems.append(f"this logo artwork is for light fields but sits on a dark one (luminance {solid['mean_lum']:.0f})")
        if problems:
            n_bad += 1
            rep.issues.append(_issue(IssueCategory.logo_violation, f"{el.id}: " + "; ".join(problems), [el.id],
                                     "fix the placement; brand logos are never altered"))
    if "logo" in brand.mandatory_elements and not any(e.role == "logo" for e in spec.placed):
        n_bad += 1
        rep.issues.append(_issue(IssueCategory.logo_violation, "the poster has no institution logo (every poster must carry it)", [],
                                 "add the brand logo"))
    rep.results.append(CheckResult("logo_rules", n_bad == 0, f"{len(logos)} logo(s) checked: {n_bad} with violations"))
