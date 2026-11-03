import copy
import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from poster_agent.assets import ComfyError
from poster_agent.compile.vc_client import cli_path
from poster_agent.ingest import load_brief, scan_assets
from poster_agent.llm import ScriptedLLM
from poster_agent.pipeline import make_poster
from poster_agent.spec.brand import ROOT, load_brand

needs_vc = pytest.mark.skipif(not cli_path().exists(), reason="vectorcraft-cli not installed")
BRIEF_DIR = ROOT / "briefs/kciiri_annual_day"
GOOD_PLAN = json.loads((Path(__file__).parent / "data/kciiri_plan_llm.json").read_text(encoding="utf-8"))
KCT = load_brand("kct")

ART_PLAN = copy.deepcopy(GOOD_PLAN)
ART_PLAN["background"] = {"kind": "generated", "color": "surface", "art": {
    "description": "translucent navy glass panels with soft gradients and a faint dotted grid", "tone": "light",
    "palette": ["primary", "accent"]}}
for _c in ART_PLAN["components"]:  # brand logos need a flat field: on generated art the logo gets a plate
    if _c["component"] == "LogoBlock":
        _c["content"]["plate"] = True


class FakeComfy:
    """Stands in for the GPU PC: returns a noisy PNG and records the workflows it was asked to run."""

    def __init__(self, fail: bool = False):
        self.fail, self.runs, self.freed = fail, [], 0

    def run(self, workflow: dict, save_node: str) -> bytes:
        if self.fail:
            raise ComfyError("ComfyUI offline")
        self.runs.append(workflow)
        buf = BytesIO()
        Image.effect_noise((912, 1140), 30).convert("RGB").save(buf, "PNG")
        return buf.getvalue()

    def free_vram(self) -> None:
        self.freed += 1


@pytest.fixture
def brief_inv():
    return load_brief(BRIEF_DIR / "brief.yaml"), scan_assets(BRIEF_DIR / "assets")


@needs_vc
def test_generated_background_is_made_around_the_text(brief_inv, proj_tmp):
    brief, inv = brief_inv
    comfy = FakeComfy()
    res = make_poster(brief, KCT, inv, ScriptedLLM([ART_PLAN]), proj_tmp / "run", comfy=comfy, cache_dir=proj_tmp / "cache", checks=False)
    assert res.ok and res.compiled.overflow == []
    assert res.spec.background.kind == "generated" and res.spec.assets["background"].provenance == "generated"
    info = res.generated[0]
    assert "open space" in info.prompt  # the layout's text zones reached the image prompt, as a described scene
    assert "main visual interest sits" in info.prompt and "colour palette is navy" in info.prompt
    assert info.seed == res.spec.assets["background"].seed  # reproducible: seed persisted on the asset
    report = json.loads((proj_tmp / "run/report.json").read_text(encoding="utf-8"))
    assert report["ok"] and report["generated"][0]["model"] == "flux3_klein"
    assert Image.open(res.compiled.png).size == (2160, 2700)


@needs_vc
def test_failed_generation_falls_back_honestly(brief_inv, proj_tmp):
    brief, inv = brief_inv
    res = make_poster(brief, KCT, inv, ScriptedLLM([ART_PLAN]), proj_tmp / "run", comfy=FakeComfy(fail=True),
                      cache_dir=proj_tmp / "cache", checks=False)
    assert res.ok and res.spec.background.kind == "solid"
    assert any("generated background failed" in n for n in res.notes)
    assert any(i.severity == "warning" and "solid ground used" in i.evidence for i in res.spec.issues)


@needs_vc
def test_without_a_generator_the_planner_is_told_and_replans(brief_inv, proj_tmp):
    brief, inv = brief_inv
    llm = ScriptedLLM([ART_PLAN, GOOD_PLAN])
    res = make_poster(brief, KCT, inv, llm, proj_tmp / "run", comfy=None, checks=False)
    assert res.ok and res.plan.rounds[0].stage == "validate"
    assert "not available" in llm.calls[1]["user"]
    assert res.spec.background.kind == "solid"


def test_planning_failure_halts_and_reports(brief_inv, proj_tmp):
    brief, inv = brief_inv
    bad = copy.deepcopy(GOOD_PLAN)
    bad["components"] = [c for c in bad["components"] if c["component"] != "LogoBlock"]
    res = make_poster(brief, KCT, inv, ScriptedLLM([bad, bad, bad]), proj_tmp / "run", comfy=None)
    assert not res.ok and res.compiled is None
    report = json.loads((proj_tmp / "run/report.json").read_text(encoding="utf-8"))
    assert report["ok"] is False and report["problems"] and report["outputs"] is None
