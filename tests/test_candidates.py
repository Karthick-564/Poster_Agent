"""Art analysis and art-aware placement: text goes where the picture is calm and away from its subject."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from poster_agent.analysis import analyse_image
from poster_agent.layout.candidates import apply_choice, choose_layout, score
from poster_agent.layout.solver import solve_layout
from poster_agent.spec import CanvasSpec, ComponentSpec, PosterSpec
from poster_agent.spec.brand import load_brand

KCT = load_brand("kct")
W, H = 1080.0, 1350.0


def art_with_subject_on_the_right() -> Image.Image:
    """Smooth pale ground, a busy detailed subject filling the right third of the picture."""
    rng = np.random.default_rng(1)
    base = Image.new("RGB", (540, 675), (225, 232, 228))
    noise = Image.fromarray(rng.integers(0, 255, (675, 180, 3), dtype=np.uint8)).filter(ImageFilter.GaussianBlur(1))
    base.paste(noise, (360, 0))
    ImageDraw.Draw(base).ellipse((330, 200, 540, 480), fill=(20, 90, 40))
    return base


def spec_with_text(**kw) -> PosterSpec:
    comps = [ComponentSpec(id="t", component="TitleStack", content={"title": "GREEN CAMPUS TALK", "align": "left"}, emphasis="hero"),
             ComponentSpec(id="d", component="TextBlock", content={"lines": ["THURSDAY, 4 MARCH 2027", "AUDITORIUM"], "align": "left"})]
    return PosterSpec(brief_id="t", brand="kct", canvas=CanvasSpec.from_format("fourfive"), components=comps, align="left", **kw)


def test_busy_and_subject_maps_find_the_detailed_right_side():
    from poster_agent.spec import Box
    art = analyse_image(art_with_subject_on_the_right(), W, H)
    left, right = Box(x=40, y=300, w=400, h=500), Box(x=700, y=300, w=340, h=500)
    assert art.busy_in(right) > art.busy_in(left) + 0.2
    assert art.subject_in(right) > art.subject_in(left)
    cx = art.subject_centroid()
    assert cx is not None and cx[0] > W * 0.55


def test_text_is_placed_away_from_the_subject_and_the_choice_is_deterministic():
    art = analyse_image(art_with_subject_on_the_right(), W, H)
    spec = spec_with_text(flow_region=(0.0, 0.0, 1.0, 1.0))
    one, two = choose_layout(spec, KCT, art), choose_layout(spec, KCT, art)
    assert (one.chosen.region, one.chosen.valign) == (two.chosen.region, two.chosen.valign)
    apply_choice(spec, one)
    res = solve_layout(spec, KCT)
    cost, terms = score(res.placed, art)
    assert terms["subject"] < 0.15 and terms["busy"] < 0.3  # the text sits on the calm side
    assert max(e.box.right for e in res.placed if e.kind.value == "text") < W * 0.78  # and stays out of the right third


def test_a_calm_picture_keeps_the_planners_placement():
    art = analyse_image(Image.new("RGB", (540, 675), (235, 235, 235)), W, H)
    spec = spec_with_text()
    choice = choose_layout(spec, KCT, art)
    assert not choice.changed and "kept as planned" in choice.note()


def test_note_reports_measured_values_for_a_move():
    art = analyse_image(art_with_subject_on_the_right(), W, H)
    choice = choose_layout(spec_with_text(), KCT, art)
    if choice.changed:
        assert "layout moved to fit the art" in choice.note() and "subject overlap" in choice.note()


def test_refinement_only_lowers_the_cost_and_stays_legal():
    from poster_agent.layout.candidates import Candidate, _try, refine
    art = analyse_image(art_with_subject_on_the_right(), W, H)
    spec = spec_with_text()
    start = _try(spec, KCT, art, (0.0, 0.0, 1.0, 1.0), "top", None)
    assert start is not None
    out = refine(spec, KCT, art, start, None)
    assert isinstance(out, Candidate) and out.cost <= start.cost
    assert _try(spec, KCT, art, out.region, out.valign, None) is not None  # still a legal layout
