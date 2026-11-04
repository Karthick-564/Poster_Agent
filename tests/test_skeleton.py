"""The house skeleton and its pieces: facts grouped on one line, labelled facts, logos in a strip, and a bottom anchor that takes the
height the content leaves so a sparse poster has no empty lower part."""
from PIL import Image

from poster_agent.checks.emptiness import planned_emptiness
from poster_agent.layout.solver import solve_layout
from poster_agent.plan.schema import PlanOutput
from poster_agent.plan.validate import validate_plan
from poster_agent.ingest.brief import Brief
from poster_agent.spec import AssetKind, AssetRef, CanvasSpec, ComponentSpec, ElementKind, PosterSpec
from poster_agent.spec.brand import ROOT, load_brand

KCT = load_brand("kct")


def cutout_asset(name="cut.png", size=(600, 800)) -> AssetRef:
    d = ROOT / "runs/_tests"
    d.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size, (90, 120, 160, 255)).save(d / name)
    return AssetRef(id=name, kind=AssetKind.generated_hero, path=(d / name).relative_to(ROOT).as_posix())


def spec(comps, template="house", assets=None, fmt="story", **kw) -> PosterSpec:
    return PosterSpec(brief_id="t", brand="kct", canvas=CanvasSpec.from_format(fmt), components=comps, template=template,
                      assets=assets or {}, **kw)


def texts(res):
    return [e for e in res.placed if e.kind == ElementKind.text]


def test_fact_line_puts_the_facts_on_one_row_with_hairlines_or_stacks_them():
    c = ComponentSpec(id="f", component="FactLine", content={"items": ["12 MARCH 2027", "10:00 AM", "AUDITORIUM"]})
    res = solve_layout(spec([c], template="classic"), KCT)
    t = texts(res)
    assert [e.text for e in t] == ["12 MARCH 2027", "10:00 AM", "AUDITORIUM"] and len({round(e.box.y) for e in t}) == 1
    assert len([e for e in res.placed if e.id.startswith("f.sep")]) == 2
    long = ComponentSpec(id="f", component="FactLine", content={"items": ["SATURDAY, 12 MARCH 2027", "10:00 AM TO 6:00 PM",
                                                                         "MAIN AUDITORIUM AND WORKSHOP BLOCK, EAST WING"]})
    res = solve_layout(spec([long], template="classic", fmt="square"), KCT)
    assert len({round(e.box.y) for e in texts(res)}) == 3 and not [e for e in res.placed if ".sep" in e.id]


def test_fact_strip_puts_small_labels_above_values_with_rules_between_columns():
    c = ComponentSpec(id="s", component="FactStrip", content={"items": [{"label": "Fee", "value": "Rs. 450"}, {"label": "Mode", "value": "MS Teams"},
                                                                        {"label": "Contact", "value": "+91 77083 91477"}]})
    res = solve_layout(spec([c], template="classic"), KCT)
    by = {e.id: e for e in res.placed}
    assert by["s.l0"].box.bottom <= by["s.v0"].box.y + 1 and by["s.l0"].size < by["s.v0"].size
    assert len([e for e in res.placed if e.id.startswith("s.sep")]) == 2


def test_the_bottom_anchor_takes_the_leftover_height_so_a_sparse_poster_has_no_empty_band():
    comps = [ComponentSpec(id="h", component="HeaderRow", content={"logo": "quadrangle", "lines": ["12 MARCH 2027", "AUDITORIUM"]}),
             ComponentSpec(id="t", component="TitleStack", content={"title": "OPEN DAY", "eyebrow": "Welcome"}, emphasis="hero"),
             ComponentSpec(id="a", component="BottomCutout", anchor="bottom", content={"asset": "cut.png", "align": "right"})]
    res = solve_layout(spec(comps, assets={"cut.png": cutout_asset()}), KCT)
    img = next(e for e in res.placed if e.id == "a.img")
    content_bottom = max(e.box.bottom for e in texts(res))
    H = CanvasSpec.from_format("story").height
    assert abs(img.box.bottom - H) < 1  # it stands on the bottom edge
    assert img.box.h <= 0.38 * H + 1  # the anchor is capped at 38%, so it never swallows the poster
    assert img.box.y - content_bottom < 0.2 * H  # and the rest of the free height is split around the content, not left as one hole
    emp = planned_emptiness(res.placed, CanvasSpec.from_format("story").width, H)
    assert emp.longest_band_fraction < 0.2


def test_split_keeps_the_text_in_the_left_column_and_the_cutout_beside_it():
    comps = [ComponentSpec(id="h", component="HeaderRow", content={"logo": "quadrangle", "lines": ["12 MARCH 2027"]}),
             ComponentSpec(id="t", component="TitleStack", content={"title": "WOES OF THE RICH", "align": "left"}, emphasis="hero"),
             ComponentSpec(id="d", component="TextBlock", content={"lines": ["A lecture on family wealth"], "align": "left"}),
             ComponentSpec(id="a", component="BottomCutout", anchor="bottom", content={"asset": "cut.png", "align": "right", "beside": True,
                                                                                         "height_pct": 45})]
    res = solve_layout(spec(comps, template="split", assets={"cut.png": cutout_asset()}, align="left"), KCT)
    W = CanvasSpec.from_format("story").width
    body = [e for e in texts(res) if e.id.split(".")[0] in ("t", "d")]
    img = next(e for e in res.placed if e.id == "a.img")
    assert max(e.box.right for e in body) <= 0.66 * W and img.box.cx > 0.6 * W


def test_skeletons_need_their_anchor_and_a_logo_strip_counts_as_the_logo():
    brief = Brief(brand="kct", event_title="OPEN DAY", format="story")
    base = {"format": "story", "background": {"kind": "solid", "color": "surface"},
            "components": [{"id": "t", "component": "TitleStack", "content": {"title": "OPEN DAY"}, "emphasis": "hero"},
                           {"id": "l", "component": "LogoStrip", "content": {}}]}
    errs = validate_plan(PlanOutput.model_validate({**base, "template": "house"}), brief, KCT, {})
    assert any("anchor at the bottom edge" in e for e in errs) and not any("exactly one institution logo" in e for e in errs)
    errs = validate_plan(PlanOutput.model_validate({**base, "template": "split"}), brief, KCT, {})
    assert any("beside: true" in e for e in errs)


def test_the_bottom_anchor_shrinks_before_the_content_has_to_fail_to_fit():
    comps = [ComponentSpec(id="h", component="HeaderRow", content={"logo": "quadrangle", "lines": ["12 MARCH 2027"]}),
             ComponentSpec(id="t", component="TextBlock", content={"lines": [f"LINE {i} OF A DENSE OUTLINE THAT NEEDS ROOM" for i in range(14)]}),
             ComponentSpec(id="a", component="BottomCutout", anchor="bottom", content={"asset": "cut.png", "height_pct": 40})]
    res = solve_layout(spec(comps, assets={"cut.png": cutout_asset()}), KCT)
    assert not [i for i in res.issues if i.category.value == "fit"]  # fits because the anchor gave way
    assert next(e for e in res.placed if e.id == "a.img").box.h < 0.40 * CanvasSpec.from_format("story").height
