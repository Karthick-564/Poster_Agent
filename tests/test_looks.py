"""Looks: what makes posters on the same skeleton differ, applied to the plan so it cannot be skipped."""
import copy

from poster_agent.ingest.brief import Brief
from poster_agent.layout.solver import solve_layout
from poster_agent.plan.looks import LOOKS, apply_look, look_for, tint_hex
from poster_agent.plan.schema import PlanOutput
from poster_agent.spec import CanvasSpec, ComponentSpec, ElementKind, PosterSpec
from poster_agent.spec.brand import load_brand

KCT = load_brand("kct")


def plan(art=True, hero=True) -> PlanOutput:
    comps = [{"id": "hdr", "component": "HeaderRow", "content": {"logo": "quadrangle", "lines": ["1 MARCH 2027"], "color": "primary"}},
             {"id": "title", "component": "TitleStack", "content": {"title": "DESIGNING A GREEN CAMPUS", "title_color": "primary"}, "emphasis": "hero"}]
    if hero:
        comps.append({"id": "cut", "component": "BottomCutout", "anchor": "bottom", "content": {"asset": "gear"}})
    return PlanOutput.model_validate({
        "format": "story", "template": "house",
        "background": ({"kind": "generated", "color": "primary", "art": {"description": "soft leaves", "tone": "light"}} if art
                       else {"kind": "solid", "color": "surface"}),
        "generated_assets": [{"id": "gear", "kind": "hero_cutout", "subject": "a steel gear"}] if hero else [], "components": comps})


def test_looks_differ_by_brief_but_are_stable_for_the_same_brief():
    a, b = Brief(brand="kct", event_title="GENERATIVE AI WORKSHOP"), Brief(brand="kct", event_title="DATA ANALYTICS WORKSHOP")
    assert look_for(a).name == look_for(Brief(brand="kct", event_title="GENERATIVE AI WORKSHOP")).name
    seen = {look_for(Brief(brand="kct", event_title=f"{w} WORKSHOP")).name for w in ["AI", "DATA", "ROBOTICS", "DESIGN", "FINANCE", "WRITING"]}
    assert len(seen) >= 2 and seen <= set(LOOKS)
    assert look_for(Brief(brand="kct", event_title="X", look="dark_art")).name == "dark_art"


def test_a_cutout_look_gets_a_plain_ground_and_drops_the_art_so_there_is_one_image_idea():
    p = plan(art=True, hero=True)
    notes = apply_look(p, KCT, LOOKS["clean_light"])
    assert p.background.kind == "gradient" and p.background.gradient[-1].color == "#ffffff" and p.look == "clean_light"  # a soft ground, not flat
    assert any("background art dropped" in n for n in notes) and any(c.component == "BottomCutout" for c in p.components)
    assert "matte" in p.generated_assets[0].subject  # the hero is prompted to sit on this ground


def test_an_art_look_drops_the_cutout_and_puts_the_header_on_a_brand_band():
    p = plan(art=True, hero=True)
    apply_look(p, KCT, LOOKS["dark_art"])
    assert p.background.kind == "generated" and not p.generated_assets and not any(c.component == "BottomCutout" for c in p.components)
    hdr = p.components[0]
    assert hdr.component == "LogoBlock" and hdr.content["band"] == "primary"


def test_an_art_look_without_art_falls_back_to_a_plain_look():
    p = plan(art=False)
    apply_look(p, KCT, LOOKS["dark_art"])
    assert p.look == "brand_blue" and p.background.kind == "gradient" and p.background.gradient[0].color == KCT.palette["primary"]


def test_the_title_treatment_and_text_colours_follow_the_look():
    dark, light = plan(art=False), plan(art=False)
    apply_look(dark, KCT, LOOKS["brand_blue"])
    apply_look(light, KCT, LOOKS["tint_soft"])
    td, tl = dark.components[1].content, light.components[1].content
    assert td["title_style"] == "accent_last" and td["title_color"] == "surface"
    assert tl["title_style"] == "highlight_last" and tl["title_color"] == "primary"
    assert light.background.gradient[0].color == tint_hex(KCT) and min(int(tint_hex(KCT)[i:i + 2], 16) for i in (1, 3, 5)) >= 0xe0  # a very light wash


def test_a_one_line_title_can_carry_a_treatment_and_keeps_its_words():
    for style in ("accent_last", "weight_split", "highlight_last", "accent_first"):
        c = ComponentSpec(id="t", component="TitleStack", content={"title": "KONCEPT 2027", "title_style": style, "align": "center"}, emphasis="hero")
        res = solve_layout(PosterSpec(brief_id="t", brand="kct", canvas=CanvasSpec.from_format("story"), components=[c]), KCT)
        t = [e for e in res.placed if e.kind == ElementKind.text]
        assert " ".join(e.text for e in t) == "KONCEPT 2027" and len(t) == 2, style
        assert t[0].box.right <= t[1].box.x + 1 and abs(t[0].box.y - t[1].box.y) < 1  # side by side on one baseline


def test_the_header_band_is_drawn_full_width_behind_the_header():
    c = [ComponentSpec(id="h", component="HeaderRow", content={"logo": "quadrangle", "lines": ["1 MARCH 2027"], "band": "primary", "color": "surface"}),
         ComponentSpec(id="t", component="TitleStack", content={"title": "OPEN DAY"}, emphasis="hero")]
    res = solve_layout(PosterSpec(brief_id="t", brand="kct", canvas=CanvasSpec.from_format("story"), components=c), KCT)
    band = next(e for e in res.placed if e.id == "h.band")
    W = CanvasSpec.from_format("story").width
    assert band.box.x == 0 and band.box.w == W and band.role == "info_bar" and band.props["fill"] == KCT.palette["primary"]
    assert "h.band" in res.bleed_ids


def test_a_highlight_bar_padding_may_extend_past_the_margin_without_failing_the_layout():
    c = ComponentSpec(id="t", component="TitleStack", content={"title": "DESIGNING A GREEN CAMPUS TALK FOR EVERYONE", "title_style": "highlight_last",
                                                              "align": "left"}, emphasis="hero")
    res = solve_layout(PosterSpec(brief_id="t", brand="kct", canvas=CanvasSpec.from_format("story"), components=[c], align="left"), KCT)
    assert not [i for i in res.issues if i.category.value == "margin_violation"]


def test_styled_title_lines_grow_their_frames_so_vectorcraft_never_drops_a_line_as_overflowing():
    for style in ("accent_last", "highlight_last", "weight_split"):
        for title in ("DESIGNING A GREEN CAMPUS TALK FOR EVERYONE", "KONCEPT 2027"):
            c = ComponentSpec(id="t", component="TitleStack", content={"title": title, "title_style": style, "align": "left"}, emphasis="hero")
            res = solve_layout(PosterSpec(brief_id="t", brand="kct", canvas=CanvasSpec.from_format("story"), components=[c]), KCT)
            lines = [e for e in res.placed if e.kind == ElementKind.text]
            assert len(lines) >= 2 and all(e.props.get("fit") == "autoHeight" for e in lines), (style, title)


def test_ornament_dividers_and_stripes_are_removed_and_the_hero_keeps_its_natural_colours():
    from poster_agent.plan.schema import PlanDecoration
    p = plan(art=False, hero=True)
    p.components.insert(1, ComponentSpec(id="div", component="Divider", variant="ornament"))
    p.decorations = [PlanDecoration(kind="stripe", region="right"), PlanDecoration(kind="border")]
    apply_look(p, KCT, LOOKS["clean_light"])
    assert not any(c.component == "Divider" for c in p.components) and [d.kind for d in p.decorations] == ["border"]
    subject = p.generated_assets[0].subject
    assert "natural colours" in subject and "navy" not in subject and "silver" not in subject


def test_a_cutout_beside_the_text_column_never_overlaps_it():
    from tests.test_skeleton import cutout_asset
    comps = [ComponentSpec(id="t", component="TitleStack", content={"title": "WOES OF THE RICH", "align": "left"}, emphasis="hero"),
             ComponentSpec(id="d", component="TextBlock", content={"lines": ["A lecture on family wealth and the next generation"], "align": "left"}),
             ComponentSpec(id="a", component="BottomCutout", anchor="bottom", content={"asset": "cut.png", "align": "right", "beside": True},)]
    res = solve_layout(PosterSpec(brief_id="t", brand="kct", canvas=CanvasSpec.from_format("story"), components=comps, template="split", align="left",
                                  assets={"cut.png": cutout_asset(size=(500, 800))}), KCT)
    img = next(e for e in res.placed if e.id == "a.img")
    text_right = max(e.box.right for e in res.placed if e.kind == ElementKind.text)
    assert img.box.x >= text_right - 1, (img.box.x, text_right)


def test_a_guest_card_without_a_photo_has_a_name_larger_than_the_designation_and_follows_the_poster_axis():
    c = ComponentSpec(id="g", component="ChiefGuestCard", content={"label": "CHIEF GUEST", "name": "DR. R. MEENAKSHI",
                                                                  "lines": ["DIRECTOR, INSTITUTE OF URBAN ECOLOGY"]})
    res = solve_layout(PosterSpec(brief_id="t", brand="kct", canvas=CanvasSpec.from_format("story"), components=[c], align="left"), KCT)
    by = {e.id: e for e in res.placed if e.kind == ElementKind.text}
    assert by["g.name"].size > by["g.line0"].size >= by["g.label"].size
    assert abs(by["g.name"].box.x - by["g.line0"].box.x) < 1 and not any(e.kind == ElementKind.image for e in res.placed)


def test_a_sparse_brief_on_a_plain_look_must_have_an_image_but_a_text_heavy_one_need_not():
    from poster_agent.plan.validate import validate_plan
    sparse = Brief(brand="kct", event_title="OPEN DAY", date="1 MARCH 2027", venue="HALL 1")
    heavy = Brief(brand="kct", event_title="OPEN DAY", extra_lines=[" ".join(f"word{i}" for i in range(60))])
    p = plan(art=False, hero=False)
    p.look = "clean_light"
    assert any("little text" in e for e in validate_plan(p, sparse, KCT, {}))
    assert not any("little text" in e for e in validate_plan(p, heavy, KCT, {}))
    p2 = plan(art=False, hero=True)
    p2.look = "clean_light"
    assert not any("little text" in e for e in validate_plan(p2, sparse, KCT, {}))


def test_a_header_anchored_to_the_edge_is_moved_inside_the_margins():
    p = plan(art=False, hero=False)
    p.components[0].anchor = "top"
    apply_look(p, KCT, LOOKS["clean_light"])
    assert p.components[0].anchor == "flow"
    comps = [c.model_copy(update={"anchor": "flow"}) for c in p.components]
    res = solve_layout(PosterSpec(brief_id="t", brand="kct", canvas=CanvasSpec.from_format("story"), components=comps), KCT)
    logo = next(e for e in res.placed if e.role == "logo")
    assert logo.box.x >= CanvasSpec.from_format("story").safe_box.x - 1 and logo.box.y >= CanvasSpec.from_format("story").safe_box.y - 1


def test_every_bottom_band_becomes_the_one_standard_footer():
    p = plan(art=False, hero=False)
    p.components.append(ComponentSpec(id="f", component="InfoBar", variant="band", anchor="bottom",
                                      content={"items": ["a@b.edu"], "fill": "surface", "text_color": "text", "fill_to": "accent"}))
    apply_look(p, KCT, LOOKS["clean_light"])
    f = p.components[-1]
    assert f.variant == "footer" and f.content["fill"] == "primary" and f.content["text_color"] == "surface" and "fill_to" not in f.content


def test_the_footer_has_one_text_size_whatever_the_number_of_lines_and_columns_for_several():
    def footer(items):
        c = ComponentSpec(id="f", component="InfoBar", variant="footer", anchor="bottom", content={"items": items})
        t = ComponentSpec(id="t", component="TitleStack", content={"title": "OPEN DAY"})
        res = solve_layout(PosterSpec(brief_id="t", brand="kct", canvas=CanvasSpec.from_format("story"), components=[t, c]), KCT)
        return [e for e in res.placed if e.id.startswith("f.item")], res
    one, _ = footer(["Register before 10 March"])
    three, res = footer(["Dr. S. Priya", "Seminar Hall 2", "Register by 10 March"])
    long, _ = footer(["Dr. S. Priya", "design.talk@example.edu", "Register at the department office before 10 March"])
    assert {e.size for e in one} == {e.size for e in three} == {e.size for e in long}  # the same size on every poster
    assert len({round(e.box.y) for e in three}) == 1 and len([e for e in res.placed if e.id.startswith("f.sep")]) == 2  # three columns, hairlines
    assert len({round(e.box.y) for e in long}) > 1  # a long line gets its own row instead of a smaller size
    assert any(e.id == "f.rule" for e in res.placed)  # the accent rule is the footer's signature


def test_a_supplied_person_stands_in_front_of_a_stage_arch_that_runs_behind_the_footer():
    from tests.test_skeleton import cutout_asset
    from poster_agent.spec import AssetKind, AssetRef
    d = cutout_asset("ppl2.png")
    person = AssetRef(id="ppl2.png", kind=AssetKind.headshot, path=d.path, provenance="supplied")
    comps = [ComponentSpec(id="t", component="TitleStack", content={"title": "OPEN DAY"}),
             ComponentSpec(id="a", component="BottomCutout", anchor="bottom", content={"asset": "ppl2.png"}),
             ComponentSpec(id="f", component="InfoBar", variant="footer", anchor="bottom", content={"items": ["x@y.edu"]})]
    res = solve_layout(PosterSpec(brief_id="t", brand="kct", canvas=CanvasSpec.from_format("story"), components=comps, assets={"ppl2.png": person}), KCT)
    by = {e.id: e for e in res.placed}
    assert "a.stage" in by and "a.glow" in by and by["a.stage"].z < by["f.bg"].z < by["a.img"].z  # arch, then the footer over its foot, then the person
    assert by["a.stage"].box.bottom >= CanvasSpec.from_format("story").height - 1
    gen = AssetRef(id="g.png", kind=AssetKind.generated_hero, path=d.path)
    comps[1] = ComponentSpec(id="a", component="BottomCutout", anchor="bottom", content={"asset": "g.png"})
    res = solve_layout(PosterSpec(brief_id="t", brand="kct", canvas=CanvasSpec.from_format("story"), components=comps, assets={"g.png": gen}), KCT)
    assert not any(e.id == "a.stage" for e in res.placed)  # a generated object needs no stage


def test_the_header_is_the_logo_alone_and_its_date_time_venue_move_to_the_footer():
    p = plan(art=False, hero=False)
    apply_look(p, KCT, LOOKS["clean_light"])
    assert p.components[0].component == "LogoBlock" and "lines" not in p.components[0].content
    foot = next(c for c in p.components if c.component == "InfoBar")
    assert foot.anchor == "bottom" and foot.variant == "footer" and foot.content["items"][0] == "1 MARCH 2027"
    p2 = plan(art=False, hero=False)
    p2.components.append(ComponentSpec(id="f", component="InfoBar", variant="band", anchor="bottom", content={"items": ["a@b.edu"]}))
    apply_look(p2, KCT, LOOKS["clean_light"])
    assert next(c for c in p2.components if c.component == "InfoBar").content["items"] == ["1 MARCH 2027", "a@b.edu"]  # facts first, contacts after


def test_a_guest_photo_is_a_face_in_a_circle_by_default():
    from tests.test_skeleton import cutout_asset
    from poster_agent.spec import AssetKind, AssetRef
    d = cutout_asset("g1.png")
    a = AssetRef(id="g1", kind=AssetKind.headshot, path=d.path, provenance="supplied")
    c = ComponentSpec(id="g", component="ChiefGuestCard", content={"name": "DR. A", "photo": "g1"})
    res = solve_layout(PosterSpec(brief_id="t", brand="kct", canvas=CanvasSpec.from_format("story"), components=[c], assets={"g1": a}), KCT)
    photo = next(e for e in res.placed if e.id == "g.photo")
    assert photo.props["framing"] == "face" and photo.props["crop"] == "circle"
