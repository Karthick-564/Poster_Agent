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









# ---- keep-out (brand: the logo sits on a flat field) ---------------------------------------------------------------------

def test_decoration_elements_touching_a_keepout_zone_are_dropped():
    zone = Box(x=400, y=100, w=200, h=150)
    free = expand(deco("dot_grid", region="top"), CANVAS, KCT, M)
    kept = expand(deco("dot_grid", region="top"), CANVAS, KCT, M, keepout=[zone])
    assert 0 < len(kept) < len(free)
    assert not any(hits_zone(e, zone) for e in kept)
    assert expand(deco("glow", region=Box(x=300, y=0, w=400, h=300)), CANVAS, KCT, M, keepout=[zone]) == []  # a pool of light over it




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




# ---- fill by scaling -------------------------------------------------------------------------------------------------------



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









# ---- routing + checks -------------------------------------------------------------------------------------------------------



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
    open_logo = KCT.model_copy(deep=True)
    for v in open_logo.logos.values():
        v.solid_field = False  # only artwork without a field of its own is plated
    res = solve_layout(spec, open_logo)
    plate = next(e for e in res.placed if e.role == "logo_plate")
    logo = next(e for e in res.placed if e.role == "logo")
    zone = logo.box.expand(KCT.logo_rules.clear_space_mm * MM_TO_PT)
    assert plate.box.contains(zone, tol=0.5) and plate.box.w < 0.4 * spec.canvas.width  # hugs the logo; not a full-width box
    assert plate.props["fill"] == KCT.palette["surface"] and plate.z < logo.z and logo.locked
    plain = solve_layout(spec.model_copy(update={"components": [ComponentSpec(id="logo", component="LogoBlock", content={"width_pct": 20})]}), KCT)
    assert not any(e.role == "logo_plate" for e in plain.placed)
    own = solve_layout(spec, KCT)  # KCT's logos carry their own navy field: a requested plate is ignored (no white box)
    assert not any(e.role == "logo_plate" for e in own.placed)


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
