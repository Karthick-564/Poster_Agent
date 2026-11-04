"""Structural components (BigTitle, TabBanner, Columns, ImageBlock, PhotoFade) and the border decoration - pure layout, no rendering."""
from PIL import Image

from poster_agent.layout.base import Ctx
from poster_agent.layout.decor import expand
from poster_agent.layout.fonts import TextMeasurer
from poster_agent.layout.solver import solve_layout
from poster_agent.spec import AssetKind, AssetRef, CanvasSpec, ComponentSpec, Decoration, ElementKind, PosterSpec
from poster_agent.spec.brand import ROOT, load_brand

KCT = load_brand("kct")
C = ComponentSpec
M = TextMeasurer()


def spec_with(components, assets=None, **kw) -> PosterSpec:
    return PosterSpec(brief_id="t", brand="kct", canvas=CanvasSpec.from_format("fourfive"), components=components,
                      assets=assets or {}, **kw)


def png_asset(tmp_dir, name="img.png", size=(400, 300)) -> AssetRef:
    d = ROOT / "runs/_tests"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    Image.new("RGB", size, (90, 120, 160)).save(p)
    return AssetRef(id=name, kind=AssetKind.photo, path=p.relative_to(ROOT).as_posix())


def test_big_title_spans_the_width_and_lines_do_not_overlap():
    s = spec_with([C(id="t", component="BigTitle", content={"lines": ["RECRUITMENT", "DRIVE"], "cap_pct": 30, "pill": "Management"})])
    res = solve_layout(s, KCT)
    lines = [e for e in res.placed if e.id in ("t.l0", "t.l1")]
    assert len(lines) == 2
    w = s.canvas.safe_box.w
    font = KCT.fonts["display_heavy"]
    width0 = M.width(lines[0].text, font.family, font.style, lines[0].size)
    assert 0.85 * w < width0 <= w  # the long word fills the safe width
    assert lines[1].size == lines[0].size or lines[1].size <= 0.3 * s.canvas.height  # the short word is capped, never overflows
    assert lines[0].box.bottom <= lines[1].box.y + 0.5  # boxes are disjoint
    pill = next(e for e in res.placed if e.id == "t.pillbg")
    assert pill.props["radius"] > 0 and not [i for i in res.issues if i.severity == "error"]












def test_the_default_brand_allows_structure_not_ornament():
    assert set(KCT.allowed_decorations) == {"border", "frame", "stripe"}


def test_icon_row_places_vector_icons_labels_and_rules():
    items = [{"icon": "cog", "label": "Mechanical"}, {"icon": "zap", "label": "Electrical Systems"}, {"icon": "code", "label": "Software"}]
    s = spec_with([C(id="d", component="IconRow", content={"items": items})])
    res = solve_layout(s, KCT)
    icons = [e for e in res.placed if e.role == "icon"]
    assert len(icons) == 3 and all(e.props["recolor"] == KCT.palette["primary"] for e in icons)
    assert all(res.assets[e.asset_id].path.endswith(".svg") for e in icons)
    labels = [e for e in res.placed if e.id.startswith("d.l")]
    assert [e.text.replace("\n", " ").upper() for e in labels] == ["MECHANICAL", "ELECTRICAL SYSTEMS", "SOFTWARE"]
    assert len([e for e in res.placed if e.id.startswith("d.sep")]) == 2












# ---- background-aware text ----------------------------------------------------------------------------------------------

def _art_spec(left, right, noise=False):
    """A poster whose generated background is `left` colour on the left half and `right` on the right half (or mid-grey noise)."""
    import numpy as np
    from poster_agent.spec import BackgroundSpec, PlacedElement, ElementKind, Box
    d = ROOT / "runs/_tests"
    d.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((400, 300, 3), dtype=np.uint8)
    arr[:, :150], arr[:, 150:] = left, right
    if noise:
        arr = np.random.default_rng(1).integers(90, 170, size=arr.shape).astype(np.uint8)
    Image.fromarray(arr).save(d / "adapt_bg.png")
    ref = AssetRef(id="bg", kind=AssetKind.generated_background, path="runs/_tests/adapt_bg.png", provenance="generated")
    c = CanvasSpec(format="custom", width=300, height=400, export_scale=1.0)

    def text(i, x):
        return PlacedElement(id=i, kind=ElementKind.text, role="body", box=Box(x=x, y=100, w=120, h=40), text="hello", font="Montserrat",
                             style="Regular", size=12, color="#1e2842", z=10)  # body-size text needs 4.5:1
    return PosterSpec(brief_id="t", brand="kct", canvas=c, assets={"bg": ref},
                      background=BackgroundSpec(kind="generated", asset_id="bg", color="#ffffff"),
                      placed=[text("on_dark", 10), text("on_light", 170)])


def test_text_colour_follows_the_pixels_behind_it():
    from poster_agent.layout.adapt import adapt_to_background
    s = _art_spec((20, 30, 60), (245, 245, 245))
    notes = adapt_to_background(s, KCT, ROOT)
    colour = {e.id: e.color for e in s.placed}
    assert colour["on_dark"] == KCT.palette["surface"]  # navy text on a navy half flips to the light brand colour
    assert colour["on_light"] == "#1e2842"  # already readable: untouched
    assert len(notes) == 1 and not any(e.id.startswith("adapt.") for e in s.placed)


def test_busy_midtone_art_gets_one_soft_backing_and_adaptation_is_idempotent():
    from poster_agent.layout.adapt import adapt_to_background
    s = _art_spec((128, 128, 128), (128, 128, 128), noise=True)
    adapt_to_background(s, KCT, ROOT)
    scrims = [e for e in s.placed if e.id.startswith("adapt.")]
    assert len(scrims) == 1 and scrims[0].z < 10  # the two texts share a row: ONE band, not a box each
    stops = scrims[0].props["fill"]["gradient"]["stops"]
    assert stops[0]["opacity"] == 0.0 and stops[-1]["opacity"] == 0.0 and stops[1]["opacity"] > 0.5  # feathered edges, solid middle
    assert scrims[0].box.x == 0 and scrims[0].box.w == 300  # full width: there is no box edge to misalign
    assert all(e.color in (KCT.palette["primary"], KCT.palette["surface"]) for e in s.placed if e.kind.value == "text")
    adapt_to_background(s, KCT, ROOT)
    assert len([e for e in s.placed if e.id.startswith("adapt.")]) == 1  # running it again does not stack backings




# ---- flexible composition + typography ----------------------------------------------------------------------------------











# ---- size discipline + hierarchy checks -----------------------------------------------------------------------------------

def _solved(comps, title="GENERATIVE AI WORKSHOP", **kw):
    from poster_agent.spec import MandatoryItem
    s = spec_with(comps, mandatory=[MandatoryItem(id="event_title", label="title", text=title)], **kw)
    res = solve_layout(s, KCT)
    s.placed, s.assets = res.placed, res.assets
    return s


def test_a_logo_never_exceeds_the_brand_maximum_even_if_the_planner_asks_for_more():
    s = _solved([C(id="l", component="LogoBlock", content={"width_pct": 30})])
    logo = next(e for e in s.placed if e.role == "logo")
    rules = KCT.logo_rules
    assert logo.box.w <= rules.max_width_pct / 100 * s.canvas.width + 0.5 and logo.box.h <= rules.max_height_pct / 100 * s.canvas.height + 0.5
    small = next(e for e in _solved([C(id="l", component="LogoBlock", content={"width_pct": 12})]).placed if e.role == "logo")
    assert abs(small.box.w - 0.12 * s.canvas.width) < 1  # a request inside the limit is honoured
    assert small.box.w / small.box.h == logo.box.w / logo.box.h  # proportions never change






def test_aesthetics_checks_measure_title_lead_logo_size_and_repeated_text():
    from poster_agent.checks.aesthetics import logo_proportion, measure, repeated_text, title_dominance
    from poster_agent.spec import Box, ElementKind, PlacedElement

    def txt(i, t, size, role="body"):
        return PlacedElement(id=i, kind=ElementKind.text, role=role, box=Box(x=10, y=10, w=300, h=40), text=t, font="Montserrat",
                             style="Bold", size=size, color="#000000", z=10)
    base = spec_with([C(id="t", component="TitleStack", content={"title": "GENERATIVE AI WORKSHOP"})])
    from poster_agent.spec import MandatoryItem
    base.mandatory = [MandatoryItem(id="event_title", label="t", text="GENERATIVE AI WORKSHOP")]
    ok = base.model_copy(update={"placed": [txt("title", "GENERATIVE AI WORKSHOP", 100, "title"), txt("d", "12 DECEMBER", 50)]})
    m, issues = title_dominance(ok)
    assert m["lead_ratio"] == 2.0 and not issues
    weak = base.model_copy(update={"placed": [txt("title", "GENERATIVE AI WORKSHOP", 100, "title"), txt("d", "12 DECEMBER", 80)]})
    assert [i.severity for i in title_dominance(weak)[1]] == ["warning"]  # leads by 1.25x
    tie = base.model_copy(update={"placed": [txt("title", "GENERATIVE AI WORKSHOP", 70, "title"), txt("d", "12 DECEMBER", 66)]})
    err = title_dominance(tie)[1][0]
    assert err.severity == "error" and "leads by only 1.06x" in err.evidence and "d" in err.element_ids
    split = base.model_copy(update={"placed": [txt("l0", "GENERATIVE AI", 100, "display"), txt("l1", "WORKSHOP", 100, "display"), txt("d", "x date", 40)]})
    assert title_dominance(split)[0]["title_size"] == 100  # a title split over lines is still the title
    big_logo = base.model_copy(update={"placed": [PlacedElement(id="lg", kind=ElementKind.image, role="logo", box=Box(x=0, y=0, w=400, h=400),
                                                              asset_id="a", z=10)]})
    assert logo_proportion(big_logo)[1][0].severity == "error"  # a 37% wide square logo = 11% of the area
    mid = big_logo.model_copy(update={"placed": [big_logo.placed[0].model_copy(update={"box": Box(x=0, y=0, w=330, h=330)})]})
    assert logo_proportion(mid)[1][0].severity == "warning"  # 7.5% of the area: noticeable, not fatal
    dup = base.model_copy(update={"placed": [txt("title", "CAMPUS PLACEMENT DRIVE 2027", 60, "title"), txt("pill", "2027", 30)]})
    assert repeated_text(dup)[1][0].element_ids == ["pill", "title"]
    assert measure(spec_with([])) == []  # no components (a bare list of elements): nothing to judge
    assert {r[0] for r in measure(ok)} == {"title_dominance", "repeated_text", "container_fill", "alignment"}


def test_an_oversized_panel_holding_only_a_small_logo_is_flagged_and_a_logo_in_a_panel_is_rejected():
    from poster_agent.checks.aesthetics import container_fill, measure
    s = _solved([C(id="p", component="Panel", content={"fill": "primary", "padding_pct": 6},  # (a panel the colour of the page is not drawn)
                   children=[C(id="l", component="LogoBlock", content={"width_pct": 15})]),
                 C(id="t", component="TextBlock", content={"text": "hello"})])
    m, issues = container_fill(s)
    assert issues and issues[0].severity == "error" and "fills only" in issues[0].evidence and "p.bg" in issues[0].element_ids
    full = _solved([C(id="p", component="Panel", content={"fill": "primary"}, children=[
        C(id="a", component="TextBlock", content={"lines": ["one two three four five six seven eight"] * 6, "role": "body"})])])
    assert not container_fill(full)[1]  # a panel that holds its text is fine
    assert any(r[0] == "container_fill" for r in measure(s))


def test_the_planner_validator_rejects_logos_and_big_type_inside_panels():
    from poster_agent.ingest import Brief
    from poster_agent.plan import PlanOutput, validate_plan
    plan = PlanOutput.model_validate({"format": "fourfive", "components": [
        {"id": "p", "component": "Panel", "children": [{"id": "l", "component": "LogoBlock", "content": {}},
                                                       {"id": "w", "component": "BigWord", "content": {"text": "30"}}]}]})
    errs = validate_plan(plan, Brief(brand="kct", event_title="30 YEAR REUNION"), KCT, {})
    assert any("inside a Panel" in e and "logo" in e for e in errs) and any("BigWord" in e and "Panel" in e for e in errs)




# ---- one alignment axis per poster ---------------------------------------------------------------------------------------------

def test_bands_and_logo_follow_the_poster_axis_and_sit_on_the_safe_margin():
    comps = [C(id="h", component="HeaderRow", content={"lines": ["x"], "side": "left"}),
             C(id="t", component="TextBlock", content={"text": "body", "max_lines": 1}),
             C(id="b", component="InfoBar", variant="band", anchor="bottom", content={"items": ["ONE", "TWO"]})]
    left, centre = _solved(comps, align="left"), _solved(comps)
    get = lambda s, i: next(e for e in s.placed if e.id == i)
    margin = left.canvas.safe_box.x
    assert get(left, "b.item0").align == "left" and abs(get(left, "b.item0").box.x - margin) < 0.5  # the band's text is on the margin axis
    assert get(left, "t.text").box.x == get(left, "b.item0").box.x  # ...the same axis as the body
    assert get(centre, "b.item0").align == "center"
    only_logo = [C(id="l", component="LogoBlock", content={"align": "left", "width_pct": 15})]
    assert next(e for e in _solved(only_logo).placed if e.role == "logo").box.cx == 540  # a centred poster centres its logo, whatever the plan said
    assert next(e for e in _solved(only_logo, align="left").placed if e.role == "logo").box.x == margin






def test_planner_must_put_the_logo_in_a_header_row_on_a_left_poster():
    from poster_agent.ingest import Brief
    from poster_agent.plan import PlanOutput, validate_plan
    b = Brief(brand="kct", event_title="X EVENT", date="1 JAN")
    left = PlanOutput.model_validate({"format": "fourfive", "align": "left", "components": [{"id": "l", "component": "LogoBlock", "content": {}}]})
    assert any("HeaderRow" in e for e in validate_plan(left, b, KCT, {}))
    centred = PlanOutput.model_validate({"format": "fourfive", "components": [{"id": "l", "component": "LogoBlock", "content": {}}]})
    assert not any("HeaderRow" in e for e in validate_plan(centred, b, KCT, {}))


# ---- generated art is toned down into a backdrop --------------------------------------------------------------------------------

def _vivid(size=(270, 400)):
    import numpy as np
    a = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    a[:, : size[0] // 2] = (230, 40, 30)   # saturated red
    a[:, size[0] // 2:] = (30, 200, 60)    # saturated green
    a[::20] = 255                          # hard white lines: high local detail
    return Image.fromarray(a)


def test_loud_art_is_calmed_a_lot_and_already_quiet_art_barely():
    from poster_agent.assets.treatments import art_energy, calm_art
    vivid = _vivid()
    out, info = calm_art(vivid, KCT.palette["primary"])
    assert info["strength"] >= 0.9 and out.size == vivid.size
    assert info["after"]["saturation"] < 0.6 * info["before"]["saturation"] and info["after"]["contrast"] < info["before"]["contrast"]
    soft = Image.new("RGB", (270, 400), (150, 160, 175))
    quiet, qinfo = calm_art(soft, KCT.palette["surface"])
    assert qinfo["strength"] == 0.25  # the minimum: a quiet picture is only nudged toward the brand tint
    assert art_energy(quiet)["saturation"] <= art_energy(soft)["saturation"] + 0.02




# ---- icons are chosen by meaning --------------------------------------------------------------------------------------------------

def test_icons_follow_the_label_not_the_planners_guess():
    from poster_agent.layout.icons import choose_icon
    from poster_agent.layout.structure import icon_names
    names = set(icon_names())
    pick = lambda label, hint=None: choose_icon(label, hint, names)
    assert pick("FOOTBALL", "power") == "goal" and pick("BASKETBALL", "target") == "dribbble" and pick("VOLLEYBALL") == "volleyball"
    assert pick("ATHLETICS", "activity") == "timer" and pick("KABADDI") == "hand-fist"
    assert pick("SOFTWARE AND EMBEDDED SYSTEMS") in ("cpu", "circuit-board", "terminal", "code")
    assert pick("HYDROGEN FUEL CELL SYSTEM") in ("droplets", "flame", "battery-charging")
    assert pick("ELIGIBILITY: FINAL YEAR B.E. AND B.TECH") == "graduation-cap" and pick("BRING YOUR RESUME AND ID CARD") == "file-text"
    assert pick("DANCE") != pick("DRAMA") and pick("DRAMA") == "drama" and pick("FASHION SHOW") == "shirt"
    assert pick("MANAGEMENT", "briefcase") == "briefcase"  # a hint that fits the label is kept
    assert pick("QWERTYUIOP", "rocket") == "rocket" and pick("QWERTYUIOP") == "star"  # nothing matches: the hint, else a neutral icon
    assert pick("football", "not-an-icon") == "goal"  # an invalid hint never breaks the row
    items = [{"label": "FOOTBALL", "icon": "power"}, {"label": "KABADDI"}]  # the hint is optional
    s = _solved([C(id="d", component="IconRow", content={"items": items})])
    assert {s.assets[e.asset_id].path.split("/")[-1] for e in s.placed if e.role == "icon"} == {"goal.svg", "hand-fist.svg"}


# ---- art dissolves into an edge band instead of being cut off -----------------------------------------------------------------

def _with_band(spec, y, h, fill="#ffffff", w=300, bid="footer.bg"):
    from poster_agent.spec import Box, ElementKind, PlacedElement
    spec.placed.append(PlacedElement(id=bid, kind=ElementKind.shape, role="info_bar", box=Box(x=0, y=y, w=w, h=h), z=1, props={"fill": fill}))
    return spec


def test_art_fades_into_an_opaque_footer_band_and_a_header_band_instead_of_a_hard_cut():
    from poster_agent.layout.adapt import FADE_FRACTION, adapt_to_background
    s = _with_band(_art_spec((40, 90, 60), (40, 90, 60)), 300, 100)  # a canvas 300x400 with a white footer band on the bottom quarter
    notes = adapt_to_background(s, KCT, ROOT)
    fade = next(e for e in s.placed if e.id == "adapt.fade_footer.bg")
    assert abs(fade.box.bottom - 300) < 1e-6 and abs(fade.box.h - FADE_FRACTION * 400) < 1e-6 and fade.box.w == 300  # ends exactly at the band
    stops = fade.props["fill"]["gradient"]["stops"]
    assert (stops[0]["opacity"], stops[1]["opacity"]) == (0.0, 1.0) and stops[0]["color"] == "#ffffff"  # art -> band colour
    assert fade.props["decorative"] and fade.z < -50 and any("fades into the footer band" in n for n in notes)
    head = _with_band(_art_spec((40, 90, 60), (40, 90, 60)), 0, 60, fill="#1e2842", bid="header.bg")
    adapt_to_background(head, KCT, ROOT)
    hf = next(e for e in head.placed if e.id == "adapt.fade_header.bg")
    assert abs(hf.box.y - 60) < 1e-6 and hf.props["fill"]["gradient"]["stops"][0]["opacity"] == 1.0  # opaque at the band, clear below it
    adapt_to_background(s, KCT, ROOT)
    assert len([e for e in s.placed if e.role == "adapt_fade"]) == 1  # idempotent




def test_text_just_above_the_band_is_judged_on_the_faded_art_not_the_raw_art():
    from poster_agent.layout.adapt import adapt_to_background
    s = _with_band(_art_spec((20, 30, 60), (20, 30, 60)), 300, 100)  # dark art, white band: the last stretch of art fades to white
    near = next(e for e in s.placed if e.id == "on_dark")
    near.box = near.box.model_copy(update={"y": 285, "h": 14})  # a text line wholly inside the fade zone (264..300), where the art is near-white
    near.color = "#ffffff"  # white text would read on the raw dark art but not on the near-white fade
    adapt_to_background(s, KCT, ROOT)
    assert next(e for e in s.placed if e.id == "on_dark").color != "#ffffff"


# ---- robustness of the alignment changes -----------------------------------------------------------------------------------------

def test_a_long_word_in_a_narrow_cell_shrinks_instead_of_failing_the_layout():
    items = [{"label": w} for w in ("ATHLETICS", "FOOTBALL", "BASKETBALL", "VOLLEYBALL", "KABADDI")]
    for align in ("left", "center"):
        res = solve_layout(spec_with([C(id="e", component="IconRow", content={"items": items})], align=align), KCT)
        assert not [i for i in res.issues if i.severity == "error"], [i.evidence for i in res.issues]
    from poster_agent.layout.base import Ctx
    from poster_agent.layout.fonts import TextMeasurer
    ctx = Ctx(KCT, CanvasSpec.from_format("square"), TextMeasurer(), {})
    el, _ = ctx.text("x", "caption", "SUPERCALIFRAGILISTIC", 0, 0, 60, max_lines=3)  # a word that cannot fit even at small sizes
    assert el.size <= ctx.size("caption") and any("exceeds width" in i.evidence for i in ctx.issues) or el.size < ctx.size("caption")




# ---- OCR robustness + per-aspect calming -----------------------------------------------------------------------------------------

def test_look_alike_characters_do_not_fail_a_visible_time_and_a_dropped_line_cannot_score_100():
    from poster_agent.checks.ocr import line_score, match_score
    assert match_score("11:00 AM", "THURSDAY,4MARCH2027 11:0OAM AUDITORIUM") >= 95  # OCR read the zero as the letter O
    assert match_score("ROOM 101", "R00M lOl") >= 95
    assert match_score("ANNUAL DAY", "COMPLETELY DIFFERENT TEXT") < 60  # folding does not make unrelated text match
    block = "CHIEF GUEST: DR. R. MEENAKSHI\nDIRECTOR, INSTITUTE OF URBAN ECOLOGY\nOPEN TO ALL STUDENTS AND FACULTY"
    truncated = "CHIEF GUEST: DR. R. MEENAKSHI DIRECTOR, INSTITUTE OF URBAN ECOLOGY"
    assert match_score(block, truncated) == 100  # the plain score is fooled: what was read sits inside the expected text
    assert line_score(block, truncated) < 60 and line_score(block, block.replace("\n", " ")) == 100  # the worst line decides


def test_calming_judges_each_aspect_separately_so_pale_detailed_art_keeps_its_colour():
    from poster_agent.assets.treatments import art_energy, calm_art, calm_strengths
    import numpy as np
    base = np.zeros((300, 300, 3), dtype=float)
    base[:] = (150, 175, 150)  # pale green: low saturation
    yy, xx = np.mgrid[0:300, 0:300]
    checker = (((yy // 12) + (xx // 12)) % 2)[..., None]  # fine, hard-edged detail
    img = Image.fromarray(np.clip(base + checker * 30, 0, 255).astype(np.uint8))
    k = calm_strengths(art_energy(img))
    assert k["saturation"] == 0.0 and k["detail"] > 0.3  # detail is loud, colour is not
    out, info = calm_art(img, KCT.palette["surface"])
    before, after = info["before"]["saturation"], info["after"]["saturation"]
    assert after >= 0.8 * before  # the green is not drained
    assert info["after"]["detail"] < info["before"]["detail"]  # the fine detail is softened


# ---- layout templates with named zones (B1) ----

def _tb(i, text, **kw):
    return ComponentSpec(id=i, component="TextBlock", content={"lines": [text]}, **kw)


def test_zones_place_components_in_their_regions_and_columns_sit_side_by_side():
    comps = [_tb("a", "HEAD LINE", zone="head"), _tb("b", "TITLE LINE", zone="title"),
             _tb("l", "LEFT COLUMN", zone="left"), _tb("r", "RIGHT COLUMN", zone="right")]
    res = solve_layout(spec_with(comps, template="two_col"), KCT)
    by = {e.id.split(".")[0]: e for e in res.placed if e.kind == ElementKind.text}
    ys = {k: v.box.y for k, v in by.items()}
    cx = {k: v.box.x + v.box.w / 2 for k, v in by.items()}
    assert ys["a"] < ys["b"] < ys["l"] and abs(ys["l"] - ys["r"]) < 2 and cx["l"] < cx["r"]
    assert not [i for i in res.issues if i.category.value in {"margin_violation", "text_overlap", "fit"}]


def test_stage_template_puts_the_title_over_the_art_zone_and_details_below():
    comps = [_tb("t", "THE TITLE", zone="title"), _tb("d", "details here", zone="details"), _tb("h", "head", zone="head")]
    res = solve_layout(spec_with(comps, template="stage"), KCT)
    y = {e.id.split(".")[0]: e.box.y for e in res.placed if e.kind == ElementKind.text}
    H = CanvasSpec.from_format("fourfive").height
    assert y["h"] < H * 0.2 and 0.45 * H < y["t"] < 0.75 * H and y["d"] > y["t"]




def test_validator_rejects_unknown_template_and_zone():
    from poster_agent.plan.validate import validate_plan
    from poster_agent.plan.schema import PlanOutput
    plan = PlanOutput(format="story", template="nope", components=[_tb("a", "x", zone="head")])
    from poster_agent.ingest.brief import Brief
    errs = validate_plan(plan, Brief(brand="kct", event_title="X"), KCT, {})
    assert any("template 'nope' is unknown" in e for e in errs)


# ---- pinned header, title styles, cutout heroes without a card ----

def test_a_leading_header_row_stays_at_the_top_whatever_the_valign():
    comps = [ComponentSpec(id="hdr", component="HeaderRow", content={"lines": ["12 MARCH 2027", "AUDITORIUM"], "logo": "quadrangle"}),
             _tb("body", "BODY LINE ONE")]
    spec = spec_with(comps, assets={}, flow_valign="bottom")
    top = {e.id.split(".")[0]: e.box.y for e in solve_layout(spec, KCT).placed if e.kind == ElementKind.text}
    assert top["hdr"] < 120 and top["body"] > 600  # header at the top margin, the body packed low


def test_title_style_splits_lines_with_different_colours_and_keeps_the_words():
    c = ComponentSpec(id="t", component="TitleStack", content={"title": "DESIGNING A GREEN CAMPUS TALK FOR EVERYONE", "title_style": "accent_last",
                                                              "title_color": "primary", "align": "left"}, emphasis="hero")
    els = [e for e in solve_layout(spec_with([c]), KCT).placed if e.kind == ElementKind.text]
    assert len(els) >= 2 and " ".join(e.text for e in els).split() == "DESIGNING A GREEN CAMPUS TALK FOR EVERYONE".split()
    assert els[-1].color != els[0].color and len({e.color for e in els[:-1]}) == 1


def test_a_generated_cutout_hero_is_drawn_without_a_card_and_a_photo_keeps_it():
    cut = AssetRef(id="cut", kind=AssetKind.generated_hero, path=png_asset(None, "cut.png").path)
    photo = png_asset(None, "ph.png")
    hero = lambda a: ComponentSpec(id="h", component="HeroImage", content={"asset": a, "height_pct": 20})
    rects = lambda spec: [e for e in solve_layout(spec, KCT).placed if e.kind == ElementKind.shape]
    assert len(rects(spec_with([hero("cut")], assets={"cut": cut}))) < len(rects(spec_with([hero("ph.png")], assets={"ph.png": photo})))


# ---- balanced line breaks ----

def test_balanced_wrap_evens_the_lines_and_keeps_words_and_line_count():
    m = TextMeasurer()
    text = "DESIGNING A GREEN CAMPUS FOR EVERYONE"
    greedy = m.wrap(text, "Montserrat", "Bold", 60, 560)
    bal = m.wrap_balanced(text, "Montserrat", "Bold", 60, 560)
    assert len(bal) == len(greedy) and " ".join(bal).split() == text.split()
    spread = lambda ls: max(m.width(x, "Montserrat", "Bold", 60) for x in ls) - min(m.width(x, "Montserrat", "Bold", 60) for x in ls)
    assert spread(bal) <= spread(greedy) + 1e-6
    assert all(m.width(x, "Montserrat", "Bold", 60) <= 560 for x in bal)




def test_highlight_last_puts_a_brand_bar_behind_the_last_line_with_matching_text():
    c = ComponentSpec(id="t", component="TitleStack", content={"title": "DESIGNING A GREEN CAMPUS TALK FOR EVERYONE", "title_style": "highlight_last",
                                                              "title_color": "primary", "align": "left"}, emphasis="hero")
    res = solve_layout(spec_with([c]), KCT)
    texts = [e for e in res.placed if e.kind == ElementKind.text]
    bars = [e for e in res.placed if e.id.endswith(".hl")]
    assert len(bars) == 1 and bars[0].props["fill"] == KCT.palette["primary"] and texts[-1].color != texts[0].color
    assert bars[0].box.intersects(texts[-1].box) and not bars[0].box.intersects(texts[0].box)


# ---- QR codes ----

def test_qr_code_decodes_back_to_the_url_and_is_placed_as_a_vector():
    import cv2
    import numpy as np
    import segno
    from poster_agent.layout.qr import qr_svg
    url = "https://example.edu/register"
    svg = qr_svg(url, dark="#1e2842")
    assert svg.suffix == ".svg" and "<svg" in svg.read_text(encoding="utf-8")
    png = ROOT / "runs/_tests/qr_check.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    segno.make(url, error="m").save(str(png), scale=10, border=2)
    img = cv2.imdecode(np.fromfile(str(png), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    assert cv2.QRCodeDetector().detectAndDecode(img)[0] == url  # the generator encodes the payload we gave it
    c = ComponentSpec(id="q", component="QRBlock", content={"url": url, "caption": "Scan to register", "align": "left"})
    res = solve_layout(spec_with([c]), KCT)
    qr = next(e for e in res.placed if e.kind == ElementKind.qr)
    assert abs(qr.box.w - qr.box.h) < 1e-6 and res.assets[qr.asset_id].path.endswith(".svg")
    assert any(e.text == "Scan to register" for e in res.placed if e.kind == ElementKind.text)
