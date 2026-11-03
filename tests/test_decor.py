import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from poster_agent.checks import run_checks
from poster_agent.checks.emptiness import measure_emptiness
from poster_agent.compile.compiler import CompileResult
from poster_agent.compile.vc_client import cli_path
from poster_agent.enrich import default_preset, free_bands, is_dark_ground, top_up, watermark_candidates
from poster_agent.fixtures import industry_conclave, kciiri_components
from poster_agent.ingest import load_brief, scan_assets
from poster_agent.layout.decor import MAX_DOTS, expand, expand_all, hits_zone, region_box
from poster_agent.layout.fonts import TextMeasurer
from poster_agent.layout.solver import apply_layout, solve_layout
from poster_agent.llm import ScriptedLLM
from poster_agent.pipeline import make_poster
from poster_agent.plan import PlanOutput, validate_plan
from poster_agent.router import route_issue
from poster_agent.spec import (DECORATION_KINDS, Box, CanvasSpec, ComponentSpec, Decoration, ElementKind, Issue, IssueCategory,
                               PlacedElement, PosterSpec)
from poster_agent.spec.brand import ROOT, load_brand

needs_vc = pytest.mark.skipif(not cli_path().exists(), reason="vectorcraft-cli not installed")
_ALL = ["glow", "dot_grid", "rings", "corner_shapes", "diagonal_band", "frame", "stripe", "watermark_text"]
KCT = load_brand("kct").model_copy(update={"allowed_decorations": _ALL})  # the ornament engine is tested with every kind opted in
M = TextMeasurer()
CANVAS = CanvasSpec.from_format("fourfive")
GOOD_PLAN = json.loads((Path(__file__).parent / "data/kciiri_plan_llm.json").read_text(encoding="utf-8"))
BRIEF_DIR = ROOT / "briefs/kciiri_annual_day"


def deco(kind, **kw) -> Decoration:
    return Decoration(id=f"t_{kind}", kind=kind, **({"text": "2026"} if kind == "watermark_text" else {}) | kw)


# ---- expansion -----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", DECORATION_KINDS)
def test_every_decoration_kind_expands_behind_the_content(kind):
    els = expand(deco(kind), CANVAS, KCT, M)
    assert els, kind
    for e in els:
        assert e.z < 0 and e.props["decorative"] is True and e.role == f"decor_{kind}"
        assert e.id.startswith("decor.t_" + kind)


def test_dot_grid_is_capped_and_halftone_shaped():
    els = expand(deco("dot_grid", density=3.0), CANVAS, KCT, M)
    assert 50 < len(els) <= MAX_DOTS
    sizes = {round(e.box.w, 2) for e in els}
    assert len(sizes) > 3  # dots swell toward the middle and thin at the edges


def test_watermark_needs_text_and_is_sized_to_its_region():
    assert expand(Decoration(id="w", kind="watermark_text"), CANVAS, KCT, M) == []
    (wm,) = expand(deco("watermark_text", region="center"), CANVAS, KCT, M)
    assert wm.kind == ElementKind.text and wm.text == "2026" and wm.props["opacity"] == 0.15
    assert wm.box.w == pytest.approx(region_box("center", CANVAS).w)


def test_named_regions_cover_the_expected_parts_of_the_canvas():
    top, bottom = region_box("top", CANVAS), region_box("bottom", CANVAS)
    assert top.y == 0 and bottom.bottom == pytest.approx(CANVAS.height) and top.bottom < bottom.y
    assert region_box("left", CANVAS).right < region_box("right", CANVAS).x
    explicit = Box(x=10, y=20, w=30, h=40)
    assert region_box(explicit, CANVAS) == explicit


# ---- keep-out (brand: the logo sits on a flat field) ---------------------------------------------------------------------

def test_decoration_elements_touching_a_keepout_zone_are_dropped():
    zone = Box(x=400, y=100, w=200, h=150)
    free = expand(deco("dot_grid", region="top"), CANVAS, KCT, M)
    kept = expand(deco("dot_grid", region="top"), CANVAS, KCT, M, keepout=[zone])
    assert 0 < len(kept) < len(free)
    assert not any(hits_zone(e, zone) for e in kept)
    assert expand(deco("glow", region=Box(x=300, y=0, w=400, h=300)), CANVAS, KCT, M, keepout=[zone]) == []  # a pool of light over it


def test_hits_zone_geometry():
    zone = Box(x=100, y=100, w=100, h=100)
    ring = PlacedElement(id="r", kind=ElementKind.shape, role="decor_rings", box=Box(x=0, y=0, w=400, h=400),
                         props={"shape": "ellipse", "fill": None, "stroke": "#000", "stroke_width": 1})
    assert hits_zone(ring, zone) is False  # the zone sits well inside the ring's outline: not touched
    ring2 = ring.model_copy(update={"box": Box(x=20, y=20, w=200, h=200)})  # radius 100 around (120,120): crosses the zone
    assert hits_zone(ring2, zone) is True  # the outline passes through the zone
    frame = PlacedElement(id="f", kind=ElementKind.shape, role="decor_frame", box=Box(x=20, y=20, w=960, h=1310),
                          props={"fill": None, "stroke": "#000", "stroke_width": 2})
    assert hits_zone(frame, Box(x=300, y=300, w=100, h=100)) is False and hits_zone(frame, Box(x=0, y=0, w=60, h=60)) is True
    band = PlacedElement(id="b", kind=ElementKind.shape, role="decor_diagonal_band", box=Box(x=-200, y=400, w=1400, h=60),
                         props={"rotation": -18})
    assert hits_zone(band, Box(x=500, y=420, w=50, h=50)) is True and hits_zone(band, Box(x=0, y=1200, w=50, h=50)) is False


def test_default_decoration_never_enters_a_logo_exclusion_zone():
    spec = kciiri_components.build_spec()
    apply_layout(spec, KCT)
    spec.decorations = default_preset(spec, KCT, dark=False)
    res = apply_layout(spec, KCT)
    from poster_agent.spec.formats import MM_TO_PT
    logo = next(e for e in res.placed if e.role == "logo")
    zone = logo.box.expand(KCT.logo_rules.clear_space_mm * MM_TO_PT)
    decor = [e for e in res.placed if e.props.get("decorative")]
    assert len(decor) > 50
    assert not [e.id for e in decor if hits_zone(e, zone)]


def test_decorations_are_not_reported_as_layout_problems(tmp_path):
    spec = kciiri_components.build_spec()
    apply_layout(spec, KCT)
    spec.decorations = [deco("corner_shapes"), deco("frame"), deco("rings", region="center"), deco("diagonal_band")]
    res = solve_layout(spec, KCT)
    assert [i for i in res.issues if i.severity == "error"] == []  # off-canvas bleed shapes are intended


# ---- fill by scaling -------------------------------------------------------------------------------------------------------

def test_sparse_posters_grow_their_content_instead_of_widening_gaps():
    sparse = PosterSpec(brief_id="s", brand="kct", canvas=CANVAS, components=[
        ComponentSpec(id="title", component="TitleStack", content={"title": "ROBOTICS DAY 2026", "subtitle": "Department of Mechatronics"}),
        ComponentSpec(id="info", component="InfoBar", variant="band", anchor="bottom",
                      content={"items": ["FRIDAY, 14 AUGUST 2026", "KCT SEMINAR HALL"], "fill": "primary"})])
    grown = solve_layout(sparse, KCT)
    assert grown.fits and grown.scale > 1.0
    title = next(e for e in grown.placed if e.id == "title.title")
    base = solve_layout(sparse.model_copy(update={"canvas": CANVAS}), KCT)  # same layout, deterministic
    assert title.size == next(e for e in base.placed if e.id == "title.title").size
    dense = solve_layout(industry_conclave.build_spec(), KCT)
    assert dense.scale <= 1.0  # a full poster is never inflated


# ---- emptiness metric -----------------------------------------------------------------------------------------------------

def test_emptiness_separates_flat_from_textured_and_respects_designed_blocks():
    flat = Image.new("RGB", (1080, 1350), "white")
    r = measure_emptiness(flat, 1.0)
    assert r.largest_fraction > 0.6 and r.longest_band_fraction > 0.6 and r.regions[0][1] == r.largest_fraction
    rng = np.random.default_rng(3)
    textured = Image.fromarray(rng.integers(100, 160, (1350, 1080, 3), dtype=np.uint8))
    assert measure_emptiness(textured, 1.0).empty_fraction < 0.02
    half = flat.copy()
    half.paste(Image.fromarray(rng.integers(100, 160, (675, 1080, 3), dtype=np.uint8)), (0, 0))
    mid = measure_emptiness(half, 1.0)
    assert 0.2 < mid.empty_fraction < 0.6
    filled = measure_emptiness(flat, 1.0, [Box(x=0, y=0, w=1080, h=1350)])  # a designed block (panel/photo) is not dead space
    assert filled.empty_fraction < 0.05


def test_emptiness_reports_where_the_dead_regions_are():
    img = Image.new("RGB", (1080, 1350), "white")
    rng = np.random.default_rng(5)
    img.paste(Image.fromarray(rng.integers(0, 255, (1350, 500, 3), dtype=np.uint8)), (0, 0))  # content on the left only
    r = measure_emptiness(img, 1.0)
    box, frac = r.regions[0]
    assert box.x > 450 and box.w > 400 and frac > 0.2


# ---- enrichment -----------------------------------------------------------------------------------------------------------

def test_default_preset_respects_the_brand_allow_list_and_ground_tone():
    spec = kciiri_components.build_spec()
    apply_layout(spec, KCT)
    kinds = {d.kind for d in default_preset(spec, KCT, dark=False)}
    assert kinds == {"corner_shapes", "dot_grid", "stripe"}
    limited = KCT.model_copy(update={"allowed_decorations": ["glow"]})
    assert default_preset(spec, limited, dark=False) == []
    light, dark = default_preset(spec, KCT, False), default_preset(spec, KCT, True)
    assert next(d for d in light if d.kind == "corner_shapes").color == "primary"
    assert next(d for d in dark if d.kind == "corner_shapes").color == "surface"
    assert all(d.source == "auto" for d in light)


def test_free_bands_find_text_free_stretches_tallest_first():
    spec = kciiri_components.build_spec()
    apply_layout(spec, KCT)
    bands = free_bands(spec)
    assert bands and all(b.w == spec.canvas.width for b in bands)
    assert [b.h for b in bands] == sorted((b.h for b in bands), reverse=True)
    for b in bands:  # a free band really has no text in it
        assert not any(e.kind == ElementKind.text and e.box.intersection_area(b) > 0 for e in spec.placed if not e.props.get("decorative"))


def test_top_up_varies_the_kind_each_round_and_targets_the_regions():
    spec = kciiri_components.build_spec()
    regions = [Box(x=600, y=700, w=300, h=400), Box(x=50, y=1000, w=200, h=200)]
    first = top_up(spec, KCT, regions, dark=False, step=0)
    kinds1 = [d.kind for d in spec.decorations]
    top_up(spec, KCT, regions, dark=False, step=1)
    kinds2 = [d.kind for d in spec.decorations][len(kinds1):]
    assert len(first) == 2 and kinds1 != kinds2  # a different decoration each round
    assert all(isinstance(d.region, Box) for d in spec.decorations) and spec.decorations[0].region == regions[0]
    assert len({d.id for d in spec.decorations}) == len(spec.decorations)


def test_watermark_candidates_and_dark_ground_detection():
    assert watermark_candidates(["ANNUAL DAY 2026 & RE-LAUNCH", "FRIDAY, 14 AUGUST 2026", "Founded 1984"]) == ["2026", "1984"]
    spec = kciiri_components.build_spec()
    assert not is_dark_ground(spec, KCT) and is_dark_ground(spec, KCT, "dark")
    spec.background.kind, spec.background.color = "solid", "#0b1020"
    assert is_dark_ground(spec, KCT)


# ---- routing + checks -------------------------------------------------------------------------------------------------------

def test_dead_space_routes_to_enrich_then_escalates():
    spec = kciiri_components.build_spec()
    issue = Issue(category=IssueCategory.dead_space, evidence="x", source="check")
    assert route_issue(issue, spec, {}) == "enrich" and route_issue(issue, spec, {"enrich": 1}) == "enrich"
    assert route_issue(issue, spec, {"enrich": 2}) == "plan"


def _compiled(tmp_path, img, placed):
    png = tmp_path / "p.png"
    img.save(png)
    pdf, vcf = tmp_path / "p.pdf", tmp_path / "p.vectorcraft"
    pdf.write_bytes(b"x")
    vcf.write_bytes(b"x")
    spec = PosterSpec(brief_id="t", brand="kct", canvas=CanvasSpec(format="square", width=1000, height=1000, export_scale=1.0), placed=placed)
    return spec, CompileResult(png=png, pdf=pdf, vectorcraft=vcf, bounds={e.id: e.box for e in placed})


def test_runner_flags_dead_space_with_regions_and_exempts_decorative_text(tmp_path):
    spec, compiled = _compiled(tmp_path, Image.new("RGB", (1000, 1000), "white"), [
        PlacedElement(id="decor.w.0", kind=ElementKind.text, role="decor_watermark_text", text="2026", font="Montserrat", style="Bold",
                      size=300, color="#eeeeee", box=Box(x=100, y=300, w=800, h=380), z=-59, props={"decorative": True, "opacity": 0.1})])
    rep = run_checks(spec, KCT, compiled, use_ocr=False)
    dead = [i for i in rep.issues if i.category == IssueCategory.dead_space]
    assert dead and dead[0].severity == "error"
    emp = next(r for r in rep.results if r.name == "emptiness")
    assert not emp.passed and emp.measured["regions"] and emp.measured["empty_fraction"] > 0.5
    # the near-invisible watermark is not held to contrast/legibility rules
    assert not [i for i in rep.issues if i.category in (IssueCategory.low_contrast, IssueCategory.mandatory_missing)]
    assert next(r for r in rep.results if r.name == "contrast").detail.startswith("0 text elements")


# ---- planner ----------------------------------------------------------------------------------------------------------------

@pytest.fixture
def setup():
    return load_brief(BRIEF_DIR / "brief.yaml"), scan_assets(BRIEF_DIR / "assets")


def errs(plan, setup):
    brief, inv = setup
    return validate_plan(PlanOutput.model_validate(plan), brief, KCT, inv)


def test_planner_decorations_are_validated(setup):
    plan = copy.deepcopy(GOOD_PLAN)
    plan["decorations"] = [{"kind": "dot_grid", "region": "top", "color": "accent", "opacity": 0.2},
                           {"kind": "watermark_text", "region": "center", "color": "primary", "opacity": 0.06, "text": "2026"}]
    assert errs(plan, setup) == []
    cases = [({"kind": "confetti"}, "kind must be one of"), ({"kind": "glow", "region": "middle"}, "region"),
             ({"kind": "glow", "color": "hot_pink"}, "colour"), ({"kind": "glow", "opacity": 0.9}, "opacity"),
             ({"kind": "watermark_text", "text": "FREE LUNCH"}, "verbatim"), ({"kind": "watermark_text"}, "verbatim"),
             ({"kind": "glow", "text": "2026"}, "only watermark_text takes text")]
    for dec, expect in cases:
        bad = copy.deepcopy(GOOD_PLAN)
        bad["decorations"] = [dec]
        assert any(expect in e for e in errs(bad, setup)), (dec, errs(bad, setup))
    bad = copy.deepcopy(GOOD_PLAN)
    bad["decorations"] = [{"kind": "glow"}] * 5
    assert any("at most 4" in e for e in errs(bad, setup))


def test_planned_decorations_reach_the_spec_and_the_brand_can_restrict_them(setup):
    from poster_agent.plan import plan_poster
    brief, inv = setup
    plan = copy.deepcopy(GOOD_PLAN)
    plan["decorations"] = [{"kind": "rings", "region": "bottom", "color": "accent", "opacity": 0.15}]
    res = plan_poster(brief, KCT, inv, ScriptedLLM([plan]))
    assert res.ok and [(d.kind, d.source) for d in res.spec.decorations] == [("rings", "plan")]
    assert any(e.role == "decor_rings" for e in res.spec.placed)
    limited = KCT.model_copy(update={"allowed_decorations": ["glow"]})
    assert any("kind must be one of ['glow']" in e for e in validate_plan(PlanOutput.model_validate(plan), brief, limited, inv))


# ---- pipeline ---------------------------------------------------------------------------------------------------------------

@needs_vc
def test_pipeline_decorates_plain_posters_by_default_and_can_be_switched_off(setup, proj_tmp):
    brief, inv = setup
    on = make_poster(brief, KCT, inv, ScriptedLLM([GOOD_PLAN]), proj_tmp / "on", comfy=None, checks=False)
    assert on.ok and {d.id for d in on.spec.decorations} == {"auto_corners", "auto_dots", "auto_stripe"}
    assert all(d.source == "auto" for d in on.spec.decorations) and any(e.props.get("decorative") for e in on.spec.placed)
    off = make_poster(brief, KCT, inv, ScriptedLLM([GOOD_PLAN]), proj_tmp / "off", comfy=None, checks=False, decorate=False)
    assert off.spec.decorations == []


@needs_vc
def test_planned_decorations_are_not_replaced_by_the_default(setup, proj_tmp):
    brief, inv = setup
    plan = copy.deepcopy(GOOD_PLAN)
    plan["decorations"] = [{"kind": "glow", "region": "center", "color": "accent", "opacity": 0.2}]
    res = make_poster(brief, KCT, inv, ScriptedLLM([plan]), proj_tmp / "run", comfy=None, checks=False)
    assert [d.kind for d in res.spec.decorations] == ["glow"]


# ---- elastic hero, logo plate, panels as dead space ---------------------------------------------------------------------

def _hero_spec(tmp_path, fill=True):
    hero = tmp_path / "hero.png"
    Image.new("RGBA", (400, 600), (10, 20, 30, 255)).save(hero)
    from poster_agent.spec import AssetKind, AssetRef
    content = {"asset": "hero", "height_pct": 20} | ({} if fill else {"fill": False})
    return PosterSpec(brief_id="h", brand="kct", canvas=CanvasSpec.from_format("story"),
                      assets={"hero": AssetRef(id="hero", kind=AssetKind.generated_hero, path=str(hero), provenance="generated")},
                      components=[ComponentSpec(id="title", component="TitleStack", content={"title": "ROBOTICS DAY"}),
                                  ComponentSpec(id="hero", component="HeroImage", content=content)])


def test_hero_grows_into_spare_space_unless_told_not_to(tmp_path):
    grown = solve_layout(_hero_spec(tmp_path), KCT)
    fixed = solve_layout(_hero_spec(tmp_path, fill=False), KCT)
    g = next(e for e in grown.placed if e.role == "hero")
    f = next(e for e in fixed.placed if e.role == "hero")
    assert g.box.h > 1.8 * f.box.h and g.box.h <= 0.62 * 1920 + 1  # spare space went to the hero, within the cap
    assert g.box.w / g.box.h == pytest.approx(400 / 600, rel=0.01) and g.box.w <= grown.flow_box.w + 1  # aspect kept, inside the width
    assert f.box.h == pytest.approx(0.2 * 1920 * fixed.scale ** 0.5, rel=0.02)


def test_logo_plate_hugs_the_logo_and_contains_its_exclusion_zone():
    from poster_agent.spec.formats import MM_TO_PT
    spec = PosterSpec(brief_id="l", brand="kct", canvas=CanvasSpec.from_format("fourfive"),
                      background={"kind": "solid", "color": "#1e2842"},
                      components=[ComponentSpec(id="logo", component="LogoBlock", content={"plate": True, "width_pct": 20})])
    res = solve_layout(spec, KCT)
    plate = next(e for e in res.placed if e.role == "logo_plate")
    logo = next(e for e in res.placed if e.role == "logo")
    zone = logo.box.expand(KCT.logo_rules.clear_space_mm * MM_TO_PT)
    assert plate.box.contains(zone, tol=0.5) and plate.box.w < 0.4 * spec.canvas.width  # hugs the logo; not a full-width box
    assert plate.props["fill"] == KCT.palette["surface"] and plate.z < logo.z and logo.locked
    plain = solve_layout(spec.model_copy(update={"components": [ComponentSpec(id="logo", component="LogoBlock", content={"width_pct": 20})]}), KCT)
    assert not any(e.role == "logo_plate" for e in plain.placed)


def test_a_mostly_empty_panel_is_dead_space_not_designed_space(tmp_path):
    img = Image.new("RGB", (1000, 1000), "white")
    panel = PlacedElement(id="p", kind=ElementKind.shape, role="panel", box=Box(x=50, y=50, w=900, h=500), z=1, props={"fill": "#ffffff"})
    spec, compiled = _compiled(tmp_path, img, [panel])
    rep = run_checks(spec, KCT, compiled, use_ocr=False)
    assert any(i.category == IssueCategory.dead_space for i in rep.issues)
    plate = PlacedElement(id="lp", kind=ElementKind.shape, role="logo_plate", box=Box(x=50, y=50, w=900, h=900), z=1, props={"fill": "#ffffff"})
    spec2, compiled2 = _compiled(tmp_path, img, [plate])
    assert not any(i.category == IssueCategory.dead_space for i in run_checks(spec2, KCT, compiled2, use_ocr=False).issues)


def test_planner_may_use_a_logo_plate_but_not_in_a_forbidden_ground_colour(setup):
    plan = copy.deepcopy(GOOD_PLAN)
    logo = next(c for c in plan["components"] if c["component"] == "LogoBlock")
    logo["content"]["plate"] = True
    assert errs(plan, setup) == []
    logo["content"]["plate"] = "accent"  # KCT: the teal is not a ground
    assert any("not allowed as a background/fill" in e for e in errs(plan, setup))
    logo["content"]["plate"] = "primary"
    assert errs(plan, setup) == []


def test_default_preset_puts_a_glow_behind_the_hero(tmp_path):
    spec = _hero_spec(tmp_path)
    apply_layout(spec, KCT)
    hero = next(e for e in spec.placed if e.role == "hero")
    preset = default_preset(spec, KCT, dark=True)
    glow = next(d for d in preset if d.kind == "glow")
    assert glow.region.contains(hero.box) and glow.opacity > next(d for d in default_preset(spec, KCT, False) if d.kind == "glow").opacity
    spec.decorations = preset
    res = apply_layout(spec, KCT)
    assert any(e.role == "decor_glow" and e.z < hero.z for e in res.placed)  # drawn behind the hero


def test_rounds_that_reduce_dead_space_rank_better_even_with_equal_errors():
    from poster_agent.checks.runner import CheckReport, CheckResult
    from poster_agent.pipeline import RoundRecord

    def rec(n, empty, errors=1, score=None):
        chk = CheckReport(results=[CheckResult("emptiness", False, "", {"empty_fraction": empty})])
        return RoundRecord(n, Path("."), None, "", chk, None, errors)  # type: ignore[arg-type]

    a, b = rec(0, 0.32), rec(1, 0.18)
    assert b.key < a.key and min([a, b], key=lambda r: r.key) is b
    assert rec(2, 0.5, errors=0).key < b.key  # fewer errors still beats everything


def test_top_ups_are_visible_not_faint():
    spec = kciiri_components.build_spec()
    regions = [Box(x=60, y=60, w=300, h=1340), Box(x=720, y=60, w=300, h=1340)]
    top_up(spec, KCT, regions, dark=True, step=0)  # dot_grid + rings
    assert all(d.opacity >= 0.25 for d in spec.decorations)
    top_up(spec, KCT, regions, dark=True, step=2)  # glow + diagonal_band
    by_kind = {d.kind: d.opacity for d in spec.decorations}
    assert by_kind["glow"] >= 0.28 and by_kind["diagonal_band"] >= 0.12


@needs_vc
def test_a_hero_gets_its_glow_even_when_the_planner_chose_other_decorations(setup, tmp_path, proj_tmp):
    from poster_agent.spec import AssetKind, AssetRef
    brief, inv = setup
    hero = proj_tmp / "hero.png"
    cut = Image.new("RGBA", (400, 600), (0, 0, 0, 0))  # a real cutout: opaque subject, transparent margins
    cut.paste(Image.new("RGBA", (240, 400), (200, 200, 210, 255)), (80, 100))
    cut.save(hero)
    plan = copy.deepcopy(GOOD_PLAN)
    plan["decorations"] = [{"kind": "stripe", "region": "top", "color": "accent", "opacity": 0.5}]
    plan["generated_assets"] = [{"id": "hero_arm", "subject": "a sleek robotic arm", "aspect": "1:1"}]
    plan["components"].insert(2, {"id": "hero", "component": "HeroImage", "content": {"asset": "hero_arm", "height_pct": 7}})

    class FakeComfy:
        def run(self, wf, node):
            from io import BytesIO
            buf = BytesIO()
            Image.effect_noise((912, 1140), 30).convert("RGB").save(buf, "PNG")
            return buf.getvalue()

        def free_vram(self):
            pass

    import poster_agent.assets.generate as g
    orig = g.cutout
    g.cutout = lambda data: Image.open(hero)  # skip real matting: the point is the layout rule
    try:
        res = make_poster(brief, KCT, inv, ScriptedLLM([plan]), proj_tmp / "run", comfy=FakeComfy(), checks=False,
                          cache_dir=proj_tmp / "cache")
    finally:
        g.cutout = orig
    assert res.spec is not None, [(r.stage, r.problems) for r in res.plan.rounds]
    kinds = [d.kind for d in res.spec.decorations]
    assert res.ok and kinds == ["stripe", "glow"], kinds  # the planner's choice is kept; the hero's glow is added


def test_a_panel_the_same_colour_as_the_page_does_not_hide_decorations():
    def poster(ground, fill):
        return PosterSpec(
            brief_id="p", brand="kct", canvas=CANVAS, background={"kind": "solid", "color": ground},
            decorations=[deco("dot_grid", region="center", opacity=0.3)],
            components=[ComponentSpec(id="box", component="Panel", content={"fill": fill, "padding_pct": 6}, children=[
                ComponentSpec(id="t", component="TitleStack", content={"title": "ROBOTICS DAY"})])])

    same = solve_layout(poster(KCT.palette["primary"], "primary"), KCT)
    assert not any(e.role == "panel" for e in same.placed)  # nothing opaque over the decoration
    assert any(e.role == "decor_dot_grid" for e in same.placed) and any(e.id == "t.title" for e in same.placed)  # children still laid out
    different = solve_layout(poster("#ffffff", "primary"), KCT)
    panel = next(e for e in different.placed if e.role == "panel")  # a deliberate contrasting panel is still drawn
    assert panel.z > max(e.z for e in different.placed if e.role == "decor_dot_grid")
