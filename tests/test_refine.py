import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from poster_agent.compile.compiler import CompileResult
from poster_agent.compile.vc_client import cli_path
from poster_agent.critic import PASS_SCORE, critique
from poster_agent.fixtures import kciiri_components
from poster_agent.ingest import load_brief, scan_assets
from poster_agent.layout.solver import apply_layout
from poster_agent.llm import LLMError, ScriptedLLM
from poster_agent.pipeline import make_poster
from poster_agent.repairs import repair_contrast
from poster_agent.router import route, route_issue
from poster_agent.spec import AssetKind, AssetRef, Box, ElementKind, ElementOverride, Issue, IssueCategory, PlacedElement
from poster_agent.spec.brand import ROOT, load_brand

needs_vc = pytest.mark.skipif(not cli_path().exists(), reason="vectorcraft-cli not installed")
BRIEF_DIR = ROOT / "briefs/kciiri_annual_day"
GOOD_PLAN = json.loads((Path(__file__).parent / "data/kciiri_plan_llm.json").read_text(encoding="utf-8"))
KCT = load_brand("kct")


def issue(cat, ids=None, source="check", severity="error", evidence="x"):
    return Issue(category=cat, severity=severity, element_ids=ids or [], evidence=evidence, source=source)


@pytest.fixture
def spec():
    s = kciiri_components.build_spec()
    apply_layout(s, KCT)
    return s


def with_generated_bg(s):
    s.background.kind = "generated"
    s.assets["background"] = AssetRef(id="background", kind=AssetKind.generated_background, path="x.png", provenance="generated")
    return s


# ---- router ---------------------------------------------------------------------------------------------------------

def test_router_table(spec):
    r = lambda cat, **kw: route_issue(issue(cat, **kw), spec, {})  # noqa: E731
    assert r(IssueCategory.low_contrast) == "repair"
    for cat in (IssueCategory.hierarchy, IssueCategory.density, IssueCategory.component_choice, IssueCategory.spacing,
                IssueCategory.alignment, IssueCategory.fit, IssueCategory.text_overlap, IssueCategory.mandatory_missing,
                IssueCategory.logo_violation, IssueCategory.margin_violation):
        assert r(cat) == "plan", cat
    assert r(IssueCategory.render_artifact) == "compile" and r(IssueCategory.output_spec) == "compile"
    # art issues need generated art to regenerate; otherwise the design itself must change
    assert r(IssueCategory.bg_clash) == "plan" and r(IssueCategory.hero_quality) == "plan"


def test_router_sends_art_issues_to_assets_when_art_is_generated(spec):
    with_generated_bg(spec)
    assert route_issue(issue(IssueCategory.bg_clash), spec, {}) == "assets"
    assert route_issue(issue(IssueCategory.stray_text, ["background"]), spec, {}) == "assets"
    assert route_issue(issue(IssueCategory.stray_text, ["title.title"]), spec, {}) == "plan"  # text elsewhere: not the art's fault


def test_router_escalates_instead_of_retrying_forever(spec):
    low = issue(IssueCategory.low_contrast, ["x"])
    assert route_issue(low, spec, {"repair": 1}) == "plan"
    assert route_issue(low, with_generated_bg(spec), {"repair": 1}) == "assets"
    assert route_issue(issue(IssueCategory.output_spec), spec, {"compile": 1}) == "halt"
    grouped = route([low, issue(IssueCategory.hierarchy, source="critic")], spec, {})
    assert set(grouped) == {"repair", "plan"}


# ---- critic ---------------------------------------------------------------------------------------------------------

@pytest.fixture
def png(tmp_path):
    p = tmp_path / "render.png"
    Image.new("RGB", (1080, 1350), "white").save(p)
    return p


def critic_for(png, spec, reply, tmp_path):
    brief = load_brief(BRIEF_DIR / "brief.yaml")
    llm = ScriptedLLM([reply]) if isinstance(reply, dict) else reply
    return critique(png, spec, KCT, brief, None, llm, tmp_path), llm


def test_critic_parses_typed_issues_and_cites_real_elements(png, spec, tmp_path):
    reply = {"score": 6, "summary": "Logo overpowers the title.", "strengths": ["clean"],
             "issues": [{"category": "hierarchy", "severity": "error", "element_ids": ["logo.logo", "org.title"],
                         "evidence": "The logo is larger than the headline", "suggested_fix": "shrink the logo"}]}
    rep, llm = critic_for(png, spec, reply, tmp_path)
    assert rep.available and rep.score == 6 and not rep.passed
    assert rep.issues[0].category == IssueCategory.hierarchy and rep.issues[0].source == "critic"
    assert rep.issues[0].element_ids == ["logo.logo", "org.title"]
    assert llm.calls[0]["images"] and "ELEMENTS" in llm.calls[0]["user"]  # the VLM saw the image and the element ids


def test_critic_drops_what_it_cannot_use_and_says_so(png, spec, tmp_path):
    reply = {"score": 9, "issues": [
        {"category": "vibes", "evidence": "feels off"},                                  # not a category
        {"category": "spacing", "element_ids": ["ghost", "logo.logo"], "evidence": "tight"}]}  # one bad id
    rep, _ = critic_for(png, spec, reply, tmp_path)
    assert len(rep.issues) == 1 and rep.issues[0].element_ids == ["logo.logo"]
    assert any("vibes" in d for d in rep.dropped) and any("ghost" in d for d in rep.dropped)


def test_critic_pass_rule_and_score_clamp(png, spec, tmp_path):
    assert critic_for(png, spec, {"score": 9, "issues": []}, tmp_path)[0].passed
    assert not critic_for(png, spec, {"score": PASS_SCORE - 1, "issues": []}, tmp_path)[0].passed
    assert not critic_for(png, spec, {"score": 10, "issues": [{"category": "density", "severity": "error", "evidence": "crowded"}]},
                          tmp_path)[0].passed  # a high score does not excuse an error-level issue
    assert critic_for(png, spec, {"score": 99, "issues": []}, tmp_path)[0].score == 10


def test_unavailable_critic_is_reported_not_faked(png, spec, tmp_path):
    rep, _ = critic_for(png, spec, ScriptedLLM([]), tmp_path)  # raises LLMError
    assert not rep.available and rep.score is None and rep.issues == [] and not rep.passed
    assert "unavailable" in rep.note
    bad, _ = critic_for(png, spec, {"summary": "no score field"}, tmp_path)
    assert not bad.available and bad.score is None


# ---- repairs ----------------------------------------------------------------------------------------------------------

def test_contrast_repair_recolours_to_what_reads_and_persists_through_relayout(tmp_path, spec):
    png = tmp_path / "p.png"
    Image.new("RGB", (2160, 2700), "#1e2842").save(png)  # a dark poster
    el = next(e for e in spec.placed if e.id == "guest.name")
    el.color = "#1e2842"  # invisible: navy on navy
    compiled = CompileResult(png=png, pdf=None, vectorcraft=None, bounds={e.id: e.box for e in spec.placed})
    actions, unresolved = repair_contrast(spec, KCT, compiled, [issue(IssueCategory.low_contrast, ["guest.name"])])
    assert unresolved == [] and "guest.name" in actions[0]
    assert spec.overrides["guest.name"].color.lower() in ("#ffffff", KCT.palette["surface"])
    apply_layout(spec, KCT)  # the solver rebuilds `placed`...
    assert next(e for e in spec.placed if e.id == "guest.name").color == spec.overrides["guest.name"].color  # ...repair survives


def test_contrast_repair_gives_up_when_nothing_helps(tmp_path, spec):
    import numpy as np
    png = tmp_path / "p.png"  # a mid-grey noisy ground: no palette colour reaches 4.5:1 on both ends
    rng = np.random.default_rng(1)
    Image.fromarray(rng.integers(40, 215, (2700, 2160, 3), dtype=np.uint8)).save(png)
    compiled = CompileResult(png=png, pdf=None, vectorcraft=None, bounds={e.id: e.box for e in spec.placed})
    actions, unresolved = repair_contrast(spec, KCT, compiled, [issue(IssueCategory.low_contrast, ["guest.name"])])
    assert actions == [] and len(unresolved) == 1 and "guest.name" not in spec.overrides


# ---- the loop ---------------------------------------------------------------------------------------------------------

@pytest.fixture
def brief_inv():
    return load_brief(BRIEF_DIR / "brief.yaml"), scan_assets(BRIEF_DIR / "assets")


def vlm_reply(score, issues=()):
    return {"score": score, "summary": "s", "strengths": [], "issues": list(issues)}


HIERARCHY = {"category": "hierarchy", "severity": "error", "element_ids": ["logo.logo"],
             "evidence": "Logo dominates the headline", "suggested_fix": "make the event title the hero"}


@needs_vc
def test_loop_repairs_contrast_without_replanning(brief_inv, proj_tmp):
    brief, inv = brief_inv
    plan = copy.deepcopy(GOOD_PLAN)
    next(c for c in plan["components"] if c["component"] == "TitleStack")["content"]["title_color"] = "surface"  # white on white
    llm = ScriptedLLM([plan])
    res = make_poster(brief, KCT, inv, llm, proj_tmp / "run", comfy=None, max_refine=2, check_emptiness=False)  # isolate contrast repair
    assert res.ok and res.passed and res.best_round == 1 and len(res.rounds) == 2
    assert any("recoloured" in a for a in res.rounds[0].actions)
    assert not res.rounds[0].passed and res.rounds[1].passed
    assert len(llm.calls) == 1  # fixed deterministically: the planner was not called again
    assert res.spec.overrides  # and the repair is recorded on the final spec


@needs_vc
def test_loop_replans_from_critic_feedback_and_keeps_the_better_round(brief_inv, proj_tmp):
    brief, inv = brief_inv
    better = copy.deepcopy(GOOD_PLAN)
    next(c for c in better["components"] if c["component"] == "LogoBlock")["content"]["width_pct"] = 12
    planner = ScriptedLLM([GOOD_PLAN, better])
    vlm = ScriptedLLM([vlm_reply(5, [HIERARCHY]), vlm_reply(9)])
    res = make_poster(brief, KCT, inv, planner, proj_tmp / "run", comfy=None, checks=False, vlm=vlm, max_refine=2)
    assert res.passed and res.reviewed and res.best_round == 1 and res.critic.score == 9
    assert any(a.startswith("replanned") for a in res.rounds[0].actions)
    assert "Logo dominates the headline" in planner.calls[1]["user"]  # the critic's evidence reached the planner
    report = json.loads((proj_tmp / "run/report.json").read_text(encoding="utf-8"))
    assert [r["critic_score"] for r in report["refine_rounds"]] == [5, 9] and report["passed"] is True
    assert (proj_tmp / "run/poster.png").exists()


@needs_vc
def test_loop_with_an_unavailable_critic_does_not_invent_a_score(brief_inv, proj_tmp):
    brief, inv = brief_inv
    res = make_poster(brief, KCT, inv, ScriptedLLM([GOOD_PLAN]), proj_tmp / "run", comfy=None, checks=False,
                      vlm=ScriptedLLM([]), max_refine=1)
    assert res.ok and res.critic is not None and not res.critic.available and res.critic.score is None
    assert not res.reviewed  # passed on checks alone, and the report must not call that a reviewed poster
    report = json.loads((proj_tmp / "run/report.json").read_text(encoding="utf-8"))
    assert report["critic"]["available"] is False and report["critic"]["score"] is None and report["reviewed"] is False


@needs_vc
def test_loop_is_capped_keeps_the_best_round_and_reports_failure_honestly(brief_inv, proj_tmp):
    brief, inv = brief_inv
    planner = ScriptedLLM([GOOD_PLAN, GOOD_PLAN, GOOD_PLAN])
    vlm = ScriptedLLM([vlm_reply(4, [HIERARCHY]), vlm_reply(6, [HIERARCHY]), vlm_reply(5, [HIERARCHY])])
    res = make_poster(brief, KCT, inv, planner, proj_tmp / "run", comfy=None, checks=False, vlm=vlm, max_refine=2)
    assert len(res.rounds) == 3 and not res.passed
    assert res.best_round == 1 and res.critic.score == 6  # all rounds carry one error; the highest score wins the tie
    report = json.loads((proj_tmp / "run/report.json").read_text(encoding="utf-8"))
    assert report["passed"] is False and report["best_round"] == 1


@needs_vc
def test_loop_reverts_a_worse_revision_and_warns_the_planner(brief_inv, proj_tmp):
    brief, inv = brief_inv
    planner = ScriptedLLM([GOOD_PLAN, GOOD_PLAN, GOOD_PLAN])
    vlm = ScriptedLLM([vlm_reply(6, [HIERARCHY]), vlm_reply(4, [HIERARCHY]), vlm_reply(9)])
    res = make_poster(brief, KCT, inv, planner, proj_tmp / "run", comfy=None, checks=False, vlm=vlm, max_refine=3)
    assert res.passed and res.best_round == 2 and res.critic.score == 9
    assert any("reverted to round 0" in a for a in res.rounds[1].actions)
    assert "made it worse" in planner.calls[2]["user"]  # the third plan call knew the second attempt regressed
    assert any("was not better" in n for n in res.notes)
    report = json.loads((proj_tmp / "run/report.json").read_text(encoding="utf-8"))
    assert report["refine_rounds"][0]["issues"][0]["category"] == "hierarchy"  # per-round issues are recorded


@needs_vc
def test_loop_stops_when_revisions_stop_helping(brief_inv, proj_tmp):
    brief, inv = brief_inv
    planner = ScriptedLLM([GOOD_PLAN] * 6)
    vlm = ScriptedLLM([vlm_reply(6, [HIERARCHY]), vlm_reply(4, [HIERARCHY]), vlm_reply(4, [HIERARCHY]), vlm_reply(9)])
    res = make_poster(brief, KCT, inv, planner, proj_tmp / "run", comfy=None, checks=False, vlm=vlm, max_refine=5)
    assert len(res.rounds) == 3 and not res.passed and res.best_round == 0  # two non-improving revisions in a row => stop
    assert any("did not improve" in n for n in res.notes)


def test_only_the_implicated_assets_are_regenerated(spec):
    from poster_agent.router import implicated_assets
    hero = AssetRef(id="gen_hero", kind=AssetKind.generated_hero, path="x.png", provenance="generated")
    spec.assets["gen_hero"] = hero
    spec.placed.append(PlacedElement(id="hero.img", kind=ElementKind.image, role="hero", asset_id="gen_hero",
                                     box=Box(x=0, y=0, w=10, h=10)))
    with_generated_bg(spec)
    assert implicated_assets([issue(IssueCategory.stray_text, ["hero.img"])], spec) == {"gen_hero"}  # element id -> its asset
    assert implicated_assets([issue(IssueCategory.stray_text, ["background"])], spec) == {"background"}
    assert implicated_assets([issue(IssueCategory.hero_quality)], spec) == {"gen_hero"}
    assert implicated_assets([issue(IssueCategory.bg_clash)], spec) == {"background"}
    assert implicated_assets([issue(IssueCategory.hierarchy, ["title.title"])], spec) == set()
