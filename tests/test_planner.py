import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from poster_agent.ingest import extract_brief, load_brief, scan_assets
from poster_agent.layout.base import REGISTRY
from poster_agent.layout.catalog import CATALOG
from poster_agent.llm import FallbackLLM, LLMError, ScriptedLLM
from poster_agent.llm.base import parse_json_object, retry_delay
from poster_agent.plan import PlanOutput, plan_poster, validate_plan
from poster_agent.plan.validate import component_texts
from poster_agent.spec.brand import ROOT, load_brand

BRIEF_DIR = ROOT / "briefs/kciiri_annual_day"
GOOD_PLAN = json.loads((Path(__file__).parent / "data/kciiri_plan_llm.json").read_text(encoding="utf-8"))  # real LLM output
KCT = load_brand("kct")


@pytest.fixture
def setup():
    return load_brief(BRIEF_DIR / "brief.yaml"), scan_assets(BRIEF_DIR / "assets")


def errors_for(plan: dict, setup) -> list[str]:
    brief, inv = setup
    return validate_plan(PlanOutput.model_validate(plan), brief, KCT, inv)


def comp(plan: dict, name: str) -> dict:
    return next(c for c in plan["components"] if c["component"] == name)


def test_real_llm_plan_is_valid(setup):
    assert errors_for(GOOD_PLAN, setup) == []


def test_catalog_matches_component_registry():
    assert set(CATALOG) == set(REGISTRY)
    for name, c in REGISTRY.items():
        assert set(c.variants) == set(CATALOG[name]["variants"]), name


def test_validator_catches_invented_text(setup):
    plan = copy.deepcopy(GOOD_PLAN)
    comp(plan, "TitleStack")["content"]["tagline"] = "Register today for a free lunch"
    assert any("not in the brief" in e for e in errors_for(plan, setup))


def test_validator_catches_dropped_brief_content(setup):
    plan = copy.deepcopy(GOOD_PLAN)
    comp(plan, "InfoBar")["content"]["items"] = comp(plan, "InfoBar")["content"]["items"][:2]  # venue dropped
    assert any("venue" in e and "missing words" in e for e in errors_for(plan, setup))


def test_validator_requires_exactly_one_logo(setup):
    plan = copy.deepcopy(GOOD_PLAN)
    plan["components"] = [c for c in plan["components"] if c["component"] != "LogoBlock"]
    assert any("exactly one institution logo" in e for e in errors_for(plan, setup))


def test_validator_catches_bad_names_colours_and_assets(setup):
    plan = copy.deepcopy(GOOD_PLAN)
    plan["components"].append({"id": "x", "component": "Sparkle", "content": {}})
    comp(plan, "TitleStack")["content"]["title_color"] = "hot_pink"
    comp(plan, "ChiefGuestCard")["content"]["photo"] = "nope"
    errs = " | ".join(errors_for(plan, setup))
    assert "unknown component" in errs and "hot_pink" in errs and "neither in the inventory" in errs


def test_unused_supplied_asset_is_an_error(setup):
    plan = copy.deepcopy(GOOD_PLAN)
    plan["components"] = [c for c in plan["components"] if c["component"] != "ChiefGuestCard"]
    assert any("never used" in e for e in errors_for(plan, setup))




def test_planner_feeds_validation_errors_back_then_succeeds(setup):
    brief, inv = setup
    bad = copy.deepcopy(GOOD_PLAN)
    comp(bad, "TitleStack")["content"]["tagline"] = "Free lunch for all"
    llm = ScriptedLLM([bad, GOOD_PLAN])
    res = plan_poster(brief, KCT, inv, llm)
    assert res.ok and [r.stage for r in res.rounds] == ["validate", "ok"]
    assert "Free lunch for all" in llm.calls[1]["user"] and "PROBLEMS" in llm.calls[1]["user"]  # the error reached the model
    assert res.spec.canvas.format == "fourfive" and res.layout.fits
    assert {m.id for m in res.spec.mandatory} >= {"event_title", "venue", "logo"}


def test_planner_feeds_layout_problems_back(setup):
    brief, inv = setup
    stuffed = copy.deepcopy(GOOD_PLAN)
    stuffed["components"].insert(1, {"id": "pad", "component": "TextBlock", "emphasis": "hero",
                                     "content": {"text": brief.theme_lines[0], "role": "display", "factor": 3, "max_lines": 6}})
    llm = ScriptedLLM([stuffed, GOOD_PLAN])
    res = plan_poster(brief, KCT, inv, llm)
    assert [r.stage for r in res.rounds] == ["layout", "ok"] and res.ok
    assert "layout:" in llm.calls[1]["user"]


def test_planner_halts_honestly_when_it_cannot_converge(setup):
    brief, inv = setup
    bad = copy.deepcopy(GOOD_PLAN)
    bad["components"] = [c for c in bad["components"] if c["component"] != "LogoBlock"]
    res = plan_poster(brief, KCT, inv, ScriptedLLM([bad, bad, bad]), max_rounds=3)
    assert not res.ok and res.spec is None and res.problems


def test_schema_errors_are_reported_not_raised(setup):
    brief, inv = setup
    res = plan_poster(brief, KCT, inv, ScriptedLLM([{"nonsense": 1}, GOOD_PLAN]))
    assert res.ok and res.rounds[0].stage == "schema"


def test_extract_brief_drops_text_that_is_not_verbatim():
    source = "Annual Day 2026 on 27 July 2026 at Kumaraguru Campus. Chief guest: Dr. A. Kumar, Padma Shri Awardee."
    llm = ScriptedLLM([{"event_title": "Annual Day 2026", "date": "27 July 2026", "venue": "Kumaraguru Campus",
                        "tagline": "Innovation for tomorrow",  # invented by the model
                        "chief_guest": {"name": "Dr. A. Kumar", "lines": ["Padma Shri Awardee", "Former Director"]}}])
    brief, problems = extract_brief(source, llm, "kct")
    assert brief.event_title == "Annual Day 2026" and brief.tagline is None
    assert brief.chief_guest.lines == ["Padma Shri Awardee"]
    assert any("tagline" in p for p in problems) and any("Former Director" in p for p in problems)


def test_extract_brief_requires_a_verbatim_title():
    with pytest.raises(ValueError):
        extract_brief("Come to our event", ScriptedLLM([{"event_title": "The Grand Gala"}]), "kct")




def test_fallback_chain():
    class Down:
        name = "down"

        def generate_json(self, *a, **k):
            raise LLMError("503")

    chain = FallbackLLM([Down(), ScriptedLLM([{"ok": 1}])])
    assert chain.generate_json("s", "u") == {"ok": 1} and chain.last_used == "scripted"
    with pytest.raises(LLMError, match="all providers failed"):
        FallbackLLM([Down()]).generate_json("s", "u")




def test_fit_feedback_names_the_tallest_components_and_compact_options(setup):
    brief, inv = setup
    stuffed = copy.deepcopy(GOOD_PLAN)
    stuffed["components"].insert(1, {"id": "pad", "component": "TextBlock", "emphasis": "hero",
                                     "content": {"text": brief.theme_lines[0], "role": "display", "factor": 3, "max_lines": 6}})
    llm = ScriptedLLM([stuffed, GOOD_PLAN])
    res = plan_poster(brief, KCT, inv, llm)
    assert res.ok
    feedback = llm.calls[1]["user"]
    assert "too tall" in feedback and "tallest components:" in feedback and "pad (" in feedback  # tells the model WHICH components are the problem
    assert "InfoBar variant 'columns'" in feedback  # and what compact alternatives exist




def test_a_plan_that_only_fails_on_height_is_compacted_without_touching_text():
    from poster_agent.plan.planner import _compact
    plan = PlanOutput.model_validate({"format": "square", "components": [
        {"id": "d", "component": "Divider", "variant": "ornament", "content": {}},
        {"id": "h", "component": "HeroImage", "content": {"asset": "a", "height_pct": 40}},
        {"id": "t", "component": "BigTitle", "content": {"lines": ["HELLO"], "cap_pct": 13}}]})
    out, what = _compact(plan)
    ids = [c.id for c in out.components]
    assert ids == ["h", "t"] and out.components[0].content["height_pct"] == 22.0 and out.components[1].content["cap_pct"] == 9.0
    assert out.components[1].content["lines"] == ["HELLO"] and len(what) == 3  # text untouched; the original plan is unchanged
    assert len(plan.components) == 3
    assert _compact(out)[0] is None  # nothing left to tighten


# ---- sparse plans on plain grounds: one chance to fill the space, then the legal plan is kept ----

def _sparse_brief_and_plan():
    from poster_agent.ingest.brief import Brief
    brief = Brief(brand="kct", format="story", event_title="OPEN DAY")
    plan = {"format": "story", "background": {"kind": "solid", "color": "surface"}, "align": "left", "flow_valign": "top",
            "components": [{"id": "title", "component": "TitleStack", "content": {"title": "OPEN DAY", "align": "left"}, "emphasis": "hero"},
                           {"id": "logo", "component": "HeaderRow", "content": {"logo": "quadrangle", "lines": [], "side": "left"}}]}
    return brief, plan


def test_a_sparse_plain_plan_is_told_to_add_an_image_and_the_dead_space_estimate_flags_it():
    from poster_agent.plan.planner import _sparse_errors, build_spec
    from poster_agent.layout.solver import apply_layout
    brief, plan = _sparse_brief_and_plan()
    res = plan_poster(brief, KCT, {}, ScriptedLLM([copy.deepcopy(plan), copy.deepcopy(plan)]), max_rounds=2)
    assert not res.ok and any("little text" in p for r in res.rounds for p in r.problems)  # a sparse brief on a plain look needs an image
    spec = build_spec(PlanOutput.model_validate(plan), brief, KCT, {})
    assert any("dead_space" in e for e in _sparse_errors(spec, apply_layout(spec, KCT)))  # and the estimate says why: it would be mostly empty


def test_a_full_plan_is_not_questioned_about_empty_space(setup):
    brief, inv = setup
    res = plan_poster(brief, KCT, inv, ScriptedLLM([copy.deepcopy(GOOD_PLAN)]), max_rounds=3)
    assert res.ok and [r.stage for r in res.rounds] == ["ok"] or all("dead_space" not in p for r in res.rounds for p in r.problems)
