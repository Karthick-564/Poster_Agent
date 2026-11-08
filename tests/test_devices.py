"""The design devices (two-tone ground, pull-quote) and the concept that commits the planner to using them on purpose."""
import json
from pathlib import Path

from poster_agent.layout.solver import solve_layout
from poster_agent.plan.schema import PlanOutput
from poster_agent.plan.validate import concept_errors, panel_errors
from poster_agent.spec import BackgroundSpec, CanvasSpec, ComponentSpec, ElementKind, PosterSpec
from poster_agent.spec.brand import load_brand

KCT = load_brand("kct")
GOOD = json.loads((Path(__file__).parent / "data/kciiri_plan_llm.json").read_text(encoding="utf-8"))


def plan(**over):
    d = json.loads(json.dumps(GOOD))
    d.update(over)
    return PlanOutput.model_validate(d)


def test_a_complete_concept_passes_and_a_missing_one_is_reported():
    assert concept_errors(plan()) == []
    assert "concept is missing" in concept_errors(plan(concept=None))[0]


def test_the_concept_must_match_the_plan():
    c = dict(GOOD["concept"])
    assert any("emphasis 'hero'" in e for e in concept_errors(plan(concept={**c, "focal_point": "logo"})))
    assert any("not a component id" in e for e in concept_errors(plan(concept={**c, "focal_point": "nope"})))
    assert any("two_tone_ground" in e for e in concept_errors(plan(concept={**c, "devices": ["hero_image", "two_tone_ground"]})))
    assert any("pull_quote" in e for e in concept_errors(plan(concept={**c, "devices": ["bogus"]})))


def test_a_used_device_must_be_named():
    bg = {"kind": "solid", "color": "surface", "panel": {"color": "primary", "starts_at": "info"}}
    assert any("two_tone_ground" in e for e in concept_errors(plan(background=bg)))
    assert concept_errors(plan(background=bg, concept={**GOOD["concept"], "devices": ["hero_image", "two_tone_ground"]})) == []


def test_panel_must_start_at_a_flow_component_and_not_sit_on_art():
    assert panel_errors(plan(), KCT) == []
    bad = {"kind": "solid", "color": "surface", "panel": {"color": "primary", "starts_at": "info"}}  # info is a bottom band, not flow
    assert any("flow component" in e for e in panel_errors(plan(background=bad), KCT))


def comps(extra):
    return [ComponentSpec(id="t", component="TitleStack", content={"title": "A TRIBUTE"}, emphasis="hero"), *extra]


def test_the_ground_panel_runs_from_just_above_its_component_to_the_bottom_edge():
    spec = PosterSpec(brief_id="t", brand="kct", canvas=CanvasSpec.from_format("story"), components=comps([
        ComponentSpec(id="q", component="PullQuote", content={"quote": "Dream, dream, dream.", "attribution": "- A. P. J. Abdul Kalam"}),
        ComponentSpec(id="tx", component="TextBlock", content={"text": "Kumaraguru pays tribute"})]),
        background=BackgroundSpec(kind="solid", color="#ffffff", panel={"color": "#00305b", "starts_at": "q"}))
    res = solve_layout(spec, KCT)
    panel = next(e for e in res.placed if e.id == "ground_panel")
    q_top = min(e.box.y for e in res.placed if e.id.startswith("q."))
    assert panel.box.bottom == spec.canvas.height and panel.box.w == spec.canvas.width and panel.box.y < q_top
    assert "ground_panel" in res.bleed_ids


def test_a_pull_quote_is_the_quote_then_its_attribution_under_an_accent_rule():
    spec = PosterSpec(brief_id="t", brand="kct", canvas=CanvasSpec.from_format("story"), components=comps([
        ComponentSpec(id="q", component="PullQuote", variant="large",
                      content={"quote": "Total commitment is the common denominator", "attribution": "- Dr. A. P. J. Abdul Kalam"})]))
    res = solve_layout(spec, KCT)
    by = {e.id: e for e in res.placed}
    assert by["q.rule"].box.y < by["q.quote"].box.y < by["q.by"].box.y
    assert by["q.quote"].kind == ElementKind.text and by["q.quote"].text.startswith("Total commitment")


def test_a_panel_close_in_tone_to_the_ground_is_rejected_and_a_clear_one_passes():
    dark_ground = {"kind": "solid", "color": "primary", "panel": {"color": "primary", "starts_at": "title_stack"}}
    assert any("too close in tone" in e for e in panel_errors(plan(background=dark_ground), KCT))
    light_on_dark = {"kind": "solid", "color": "primary", "panel": {"color": "surface", "starts_at": "title_stack"}}
    assert panel_errors(plan(background=light_on_dark), KCT) == []
    dark_on_light = {"kind": "solid", "color": "surface", "panel": {"color": "primary", "starts_at": "title_stack"}}
    assert panel_errors(plan(background=dark_on_light), KCT) == []


def test_the_row_above_a_panel_keeps_clear_of_its_edge():
    def build(with_panel):
        bg = BackgroundSpec(kind="solid", color="#ffffff", panel={"color": "#00305b", "starts_at": "tx"} if with_panel else None)
        return PosterSpec(brief_id="t", brand="kct", canvas=CanvasSpec.from_format("story"), background=bg, components=comps([
            ComponentSpec(id="facts", component="TextBlock", content={"text": "22 FEBRUARY 2027"}),
            ComponentSpec(id="tx", component="TextBlock", content={"text": "ELIGIBILITY: FINAL YEAR"})]))
    res = solve_layout(build(True), KCT)
    panel = next(e for e in res.placed if e.id == "ground_panel")
    facts = next(e for e in res.placed if e.id == "facts.text")
    assert panel.box.y - facts.box.bottom >= 0.02 * 1920  # at least 2% of the poster height between the row and the panel's edge
