"""Structural components (BigTitle, TabBanner, Columns, ImageBlock, PhotoFade) and the border decoration - pure layout, no rendering."""
from PIL import Image

from poster_agent.layout.base import Ctx
from poster_agent.layout.decor import expand
from poster_agent.layout.fonts import TextMeasurer
from poster_agent.layout.solver import solve_layout
from poster_agent.spec import AssetKind, AssetRef, CanvasSpec, ComponentSpec, Decoration, PosterSpec
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


def test_tab_banner_hangs_from_the_top_and_clears_the_margin():
    s = spec_with([C(id="tab", component="TabBanner", anchor="top", content={"text": "Join us now"}),
                   C(id="b", component="TextBlock", content={"text": "hello"})])
    res = solve_layout(s, KCT)
    text = next(e for e in res.placed if e.id == "tab.text")
    bg = next(e for e in res.placed if e.id == "tab.bg")
    assert bg.box.y == 0 and text.box.y >= 0.04 * s.canvas.height  # text inside the safe margin though the tab bleeds
    body = next(e for e in res.placed if e.id == "b.text")
    assert body.box.y > bg.box.bottom  # flow content starts below the tab


def test_columns_split_rule_and_vertical_centring():
    kids = [C(id="l", component="TextBlock", content={"lines": ["a", "b", "c", "d", "e", "f"], "column": "left"}),
            C(id="r", component="TextBlock", content={"text": "x", "column": "right"})]
    s = spec_with([C(id="cols", component="Columns", content={"left_pct": 60}, children=kids)])
    res = solve_layout(s, KCT)
    left, right = (next(e for e in res.placed if e.id == i) for i in ("l.text", "r.text"))
    rule = next(e for e in res.placed if e.id == "cols.rule")
    assert left.box.right < rule.box.x < right.box.x  # a hairline sits between the columns
    assert right.box.y > left.box.y  # the short column is centred against the tall one


def test_image_block_keeps_aspect_and_gets_a_card():
    a = png_asset(None, "blk.png", (200, 100))
    s = spec_with([C(id="q", component="ImageBlock", content={"asset": a.id, "width_pct": 40})], {a.id: a})
    res = solve_layout(s, KCT)
    img = next(e for e in res.placed if e.id == "q.img")
    assert abs(img.box.w / img.box.h - 2.0) < 0.01
    assert any(e.id == "q.card" for e in res.placed)


def test_photo_fade_bleeds_full_width_and_fades_into_the_page():
    a = png_asset(None, "ph.png")
    s = spec_with([C(id="p", component="PhotoFade", anchor="bottom", content={"asset": a.id, "height_pct": 30}),
                   C(id="b", component="TextBlock", content={"text": "hello"})], {a.id: a})
    res = solve_layout(s, KCT)
    photo = next(e for e in res.placed if e.id == "p.img")
    fade = next(e for e in res.placed if e.id == "p.fade")
    assert photo.box.x == 0 and photo.box.w == s.canvas.width and photo.box.bottom == s.canvas.height
    assert photo.props["crop"] == "cover"  # cropped, never altered
    stops = fade.props["fill"]["gradient"]["stops"]
    assert stops[0]["opacity"] == 1.0 and stops[1]["opacity"] == 0.0 and fade.z > photo.z
    assert fade.box.y == photo.box.y  # the veil sits on the photo's top edge, where it meets the page
    body = next(e for e in res.placed if e.id == "b.text")
    assert body.box.bottom <= photo.box.y + 1  # flow content stays above the photo


def test_border_decoration_is_two_tone_and_behind_content():
    els = expand(Decoration(id="b", kind="border", color="primary", opacity=1.0), CanvasSpec.from_format("fourfive"), KCT, M)
    assert len(els) == 2 and all(e.props["decorative"] and e.z < 0 for e in els)
    assert els[0].props["stroke_width"] > 3 * els[1].props["stroke_width"]  # bold outer edge, fine inner keyline
    assert els[0].box.x == els[0].props["stroke_width"] / 2  # the outer edge sits on the poster edge


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


def test_icon_row_flags_unknown_icons_and_the_catalog_lists_real_ones():
    s = spec_with([C(id="d", component="IconRow", content={"items": [{"icon": "nope", "label": "X"}]})])
    assert any("unknown icon" in i.evidence for i in solve_layout(s, KCT).issues)
    from poster_agent.layout.structure import icon_names
    assert {"cog", "rocket", "graduation-cap"} <= set(icon_names())


def test_big_word_fills_width_and_caption_sits_beside_it():
    s = spec_with([C(id="w", component="BigWord", content={"text": "JOIN", "caption": "the team", "word_pct": 55})])
    res = solve_layout(s, KCT)
    word = next(e for e in res.placed if e.id == "w.word")
    cap = next(e for e in res.placed if e.id == "w.caption")
    assert word.box.right < cap.box.x and cap.box.cy == word.box.cy or abs(cap.box.cy - word.box.cy) < 2
    assert word.size > 100  # oversized


def test_ornament_divider_is_a_chain_of_diamonds():
    s = spec_with([C(id="o", component="Divider", variant="ornament", content={"count": 5})])
    res = solve_layout(s, KCT)
    d = [e for e in res.placed if e.id.startswith("o.d")]
    assert len(d) == 5 and all(e.props["rotation"] == 45 for e in d)
    assert abs(sum(e.box.cx for e in d) / 5 - s.canvas.width / 2) < 1  # centred


def test_left_composition_left_aligns_stacks_logo_and_big_type():
    comps = [C(id="logo", component="LogoBlock", content={"width_pct": 20}),
             C(id="t", component="TitleStack", content={"title": "Annual Day"}),
             C(id="b", component="BigTitle", content={"lines": ["JOIN"], "cap_pct": 8})]
    left = solve_layout(spec_with(comps, align="left"), KCT)
    centre = solve_layout(spec_with(comps), KCT)
    get = lambda r, i: next(e for e in r.placed if e.id == i)
    assert get(left, "t.title").align == "left" and get(centre, "t.title").align == "center"
    assert get(left, "logo.logo").box.x < get(centre, "logo.logo").box.x  # logo hangs on the left edge
    assert get(left, "b.l0").align == "left"


def test_duotone_is_a_brand_colour_ramp_on_a_copy_and_never_touches_logos_or_people(tmp_path):
    from poster_agent.assets.treatments import apply_treatments, duotone
    a = png_asset(None, "duo.png", (120, 80))
    head = png_asset(None, "duo_head.png", (80, 80)).model_copy(update={"id": "head", "kind": AssetKind.headshot})
    c = [C(id="p", component="PhotoFade", anchor="bottom", content={"asset": a.id, "treatment": "duotone"}),
         C(id="b", component="TextBlock", content={"text": "hi"})]
    s = spec_with(c, {a.id: a, "head": head})
    res = solve_layout(s, KCT)
    s.placed, s.assets = res.placed, res.assets
    photo = next(e for e in s.placed if e.id == "p.img")
    assert photo.props["treatment"] == "duotone"
    before = (ROOT / a.path).read_bytes()
    notes = apply_treatments(s, KCT, ROOT / "runs/_tests/treated")
    assert photo.asset_id != a.id and "applied to a copy" in notes[0]
    assert (ROOT / a.path).read_bytes() == before  # the supplied original is untouched
    t = s.assets[photo.asset_id]
    assert t.treatment["tone"] == "duotone" and t.treatment["source"] == a.path
    # a headshot asset asked to be treated is refused
    s.placed.append(photo.model_copy(update={"id": "x", "role": "photo", "asset_id": "head"}))
    assert any("never colour-treated" in n for n in apply_treatments(s, KCT, ROOT / "runs/_tests/treated"))
    # the ramp itself: darks go to the dark colour, lights to the light colour
    im = Image.new("RGB", (2, 1)); im.putpixel((0, 0), (0, 0, 0)); im.putpixel((1, 0), (255, 255, 255))
    out = duotone(im, "#102040", "#f0f4ff")
    assert out.getpixel((0, 0))[:3] == (16, 32, 64) and out.getpixel((1, 0))[:3] == (240, 244, 255)
