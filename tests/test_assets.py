import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from poster_agent.assets.generate import bg_size, cover_crop, looks_blank
from poster_agent.assets.matting import alpha_coverage
from poster_agent.assets.prompts import (ArtDirection, calm_description, calm_layout, caption_text, ideogram_prompt,
                                         qwen_subject_prompt, text_coverage_bands, validate_ideogram_caption)
from poster_agent.assets.workflows import IDEOGRAM_PRESETS, ideogram_workflow, qwen_workflow, round16
from poster_agent.ingest import load_brief, scan_assets
from poster_agent.llm import ScriptedLLM
from poster_agent.plan import PlanOutput, plan_poster, validate_plan
from poster_agent.spec import AssetKind, AssetRef, Box, CanvasSpec, ElementKind, PlacedElement
from poster_agent.spec.brand import ROOT, load_brand

BRIEF_DIR = ROOT / "briefs/kciiri_annual_day"
GOOD_PLAN = json.loads((Path(__file__).parent / "data/kciiri_plan_llm.json").read_text(encoding="utf-8"))
KCT = load_brand("kct")


def graph_refs_resolve(wf: dict) -> None:
    for name, node in wf.items():
        for key, v in node["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in wf, f"{name}.{key} references missing node {v[0]!r}"




def test_qwen_workflow_is_wired():
    wf, save = qwen_workflow("a robot arm", 1000, 1000, seed=5)
    graph_refs_resolve(wf)
    assert wf["sample"]["inputs"]["seed"] == 5 and wf[save]["class_type"] == "SaveImage"
    assert "text" in wf["neg"]["inputs"]["text"]  # negative prompt keeps printed text out of the subject




def test_blank_and_alpha_checks():
    assert looks_blank(Image.new("RGB", (64, 64), "white"))
    noisy = Image.effect_noise((64, 64), 40).convert("RGB")
    assert not looks_blank(noisy)
    rgba = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    assert alpha_coverage(rgba) == 0.0
    rgba.paste((255, 0, 0, 255), (0, 0, 5, 10))
    assert alpha_coverage(rgba) == pytest.approx(0.5)


def text_el(y, h):
    return PlacedElement(id=f"t{y}", kind=ElementKind.text, role="x", box=Box(x=0, y=y, w=1000, h=h), text="x", size=10)






# ---- planner integration with generated assets ---------------------------------------------------------------------

@pytest.fixture
def setup():
    return load_brief(BRIEF_DIR / "brief.yaml"), scan_assets(BRIEF_DIR / "assets")


def with_generated(plan: dict, subject="a sleek robotic arm in navy and teal") -> dict:
    p = copy.deepcopy(plan)
    p["generated_assets"] = [{"id": "hero_arm", "subject": subject, "aspect": "1:1"}]
    p["components"].insert(2, {"id": "hero", "component": "HeroImage", "emphasis": "primary",
                               "content": {"asset": "hero_arm", "height_pct": 7}})
    return p


def errs(plan, setup):
    brief, inv = setup
    return validate_plan(PlanOutput.model_validate(plan), brief, KCT, inv)


def test_generated_hero_validates(setup):
    assert errs(with_generated(GOOD_PLAN), setup) == []


@pytest.mark.parametrize("subject,word", [
    ("a banner reading text across it", "text"), ("a portrait of a student", "portrait"),
    ("a company logo", "logo"), ('a sign saying "hello"', "quotation")])
def test_generated_subjects_must_not_contain_text_logos_or_people(setup, subject, word):
    assert any(word in e for e in errs(with_generated(GOOD_PLAN, subject), setup))


def test_declared_generated_asset_must_be_used(setup):
    p = copy.deepcopy(GOOD_PLAN)
    p["generated_assets"] = [{"id": "orphan", "subject": "a gear"}]
    assert any("no component uses it" in e for e in errs(p, setup))


def test_generated_background_needs_clean_art(setup):
    p = copy.deepcopy(GOOD_PLAN)
    p["background"] = {"kind": "generated", "color": "surface"}
    assert any("needs background.art" in e for e in errs(p, setup))
    p["background"]["art"] = {"description": "glass panels with the word welcome in neon", "tone": "dark"}
    assert any("'word'" in e for e in errs(p, setup))
    p["background"]["art"] = {"description": "translucent navy glass panels with soft gradients", "tone": "light",
                              "palette": ["primary", "#ff00zz"]}
    assert any("#ff00zz" in e for e in errs(p, setup))
    p["background"]["art"]["palette"] = ["primary", "accent"]
    assert not any("plate" in e for e in errs(p, setup))  # KCT's logos carry their own field: no plate required
    open_logo = KCT.model_copy(deep=True)
    for v in open_logo.logos.values():
        v.solid_field = False
    brief, inv = setup
    assert any("set plate: true" in e for e in validate_plan(PlanOutput.model_validate(p), brief, open_logo, inv))
    for c in p["components"]:
        if c["component"] == "LogoBlock":
            c["content"]["plate"] = True
    assert errs(p, setup) == []


def test_planner_realizes_generated_assets_and_lays_them_out(setup, tmp_path):
    brief, inv = setup
    hero = tmp_path / "hero.png"
    Image.new("RGBA", (400, 500), (10, 20, 30, 255)).save(hero)
    calls = []

    def realize(items):
        calls.append([i.id for i in items])
        return {i.id: AssetRef(id=i.id, kind=AssetKind.generated_hero, path=str(hero), provenance="generated", seed=7,
                               prompt=i.subject) for i in items}

    res = plan_poster(brief, KCT, inv, ScriptedLLM([with_generated(GOOD_PLAN)]), realize=realize)
    assert res.ok, res.problems
    assert calls == [["hero_arm"]] and res.spec.assets["hero_arm"].seed == 7
    img = next(e for e in res.spec.placed if e.role == "hero")
    assert img.box.w / img.box.h == pytest.approx(400 / 500, rel=0.01)  # keeps its own aspect


def test_planner_reports_generation_failure_back_to_the_model(setup):
    brief, inv = setup

    def realize(items):
        raise RuntimeError("ComfyUI offline")

    llm = ScriptedLLM([with_generated(GOOD_PLAN), GOOD_PLAN])
    res = plan_poster(brief, KCT, inv, llm, realize=realize)
    assert res.ok and [r.stage for r in res.rounds] == ["assets", "ok"]
    assert "ComfyUI offline" in llm.calls[1]["user"]


def test_generated_assets_without_a_generator_are_rejected_not_ignored(setup):
    brief, inv = setup
    llm = ScriptedLLM([with_generated(GOOD_PLAN), GOOD_PLAN])
    res = plan_poster(brief, KCT, inv, llm)  # no realize hook
    assert res.ok and res.rounds[0].stage == "validate" and "not available" in llm.calls[1]["user"]


def test_default_seeds_are_derived_so_replans_hit_the_cache(tmp_path):
    from io import BytesIO

    from poster_agent.assets.generate import AssetGenerator, derived_seed

    assert derived_seed("a", "b") == derived_seed("a", "b") != derived_seed("a", "c")

    class Counting:
        runs = 0

        def run(self, wf, node):
            Counting.runs += 1
            buf = BytesIO()
            Image.effect_noise((912, 1140), 30).convert("RGB").save(buf, "PNG")
            return buf.getvalue()

        def free_vram(self):
            pass

    gen = AssetGenerator(Counting(), tmp_path / "out", tmp_path / "cache")
    art = ArtDirection(description="Soft navy glass panels.")
    canvas = CanvasSpec.from_format("fourfive")
    a, ia = gen.background("bg", art, canvas, "calm")
    b, ib = gen.background("bg", art, canvas, "calm")  # same request, e.g. after a replan
    assert Counting.runs == 1 and a.seed == b.seed and not ia.cached and ib.cached
    c, _ = gen.background("bg", art, canvas, "calm", seed=a.seed + 1)  # an explicit seed is a new draw
    assert Counting.runs == 2 and c.seed != a.seed


def test_gpu_time_is_not_spent_on_plans_that_cannot_fit(setup):
    brief, inv = setup
    stuffed = with_generated(GOOD_PLAN)
    stuffed["components"].insert(1, {"id": "pad", "component": "TextBlock", "emphasis": "hero",
                                     "content": {"text": brief.theme_lines[0], "role": "display", "factor": 3, "max_lines": 6}})
    calls = []

    def realize(items):
        calls.append([i.id for i in items])
        return {}

    llm = ScriptedLLM([stuffed, GOOD_PLAN])
    res = plan_poster(brief, KCT, inv, llm, realize=realize)
    assert res.ok and calls == [] and res.rounds[0].stage == "layout"  # the dry layout failed, so nothing was generated
    assert "layout:" in llm.calls[1]["user"]


# ---- generated images are screened for text (refusal banners, baked-in lettering) -----------------------------------------

def banner_png() -> bytes:
    from io import BytesIO

    from PIL import ImageDraw, ImageFont
    img = Image.new("RGB", (912, 1140), (110, 110, 118))
    ImageDraw.Draw(img).text((60, 500), "Image blocked by safety filter", fill=(240, 240, 240),
                             font=ImageFont.truetype(str(ROOT / "brands/fonts/Montserrat-Bold.ttf"), 62))
    buf = BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def noise_png() -> bytes:
    from io import BytesIO
    buf = BytesIO()
    Image.effect_noise((912, 1140), 30).convert("RGB").save(buf, "PNG")
    return buf.getvalue()


class Sequenced:
    def __init__(self, outputs):
        self.outputs, self.runs = list(outputs), 0

    def run(self, wf, node):
        self.runs += 1
        return self.outputs.pop(0)

    def free_vram(self):
        pass


def test_refusal_banner_is_rejected_never_cached_and_the_next_seed_is_used(tmp_path):
    from poster_agent.assets.generate import AssetGenerator
    comfy = Sequenced([banner_png(), noise_png()])
    gen = AssetGenerator(comfy, tmp_path / "out", tmp_path / "cache")
    art, canvas = ArtDirection(description="Soft navy glass panels."), CanvasSpec.from_format("fourfive")
    ref, info = gen.background("bg", art, canvas, "calm")
    assert comfy.runs == 2 and info.seed == ref.seed
    assert len(info.rejected) == 1 and "blocked" in info.rejected[0].lower()
    assert len(list((tmp_path / "cache").glob("*.png"))) == 1  # only the clean image was cached
    ref2, info2 = gen.background("bg", art, canvas, "calm")  # same request again
    assert comfy.runs == 2 and info2.cached and ref2.seed == ref.seed  # the bad seed is remembered; no GPU run at all






# ---- the official Ideogram 4 caption schema (off-schema prompts risk a false "blocked by safety filter") -------------------









# ---- FLUX 3 Klein backend (default for backgrounds) ----------------------------------------------------------------------

def test_flux_workflow_is_wired_like_the_comfyui_template():
    from poster_agent.assets.workflows import flux_workflow
    wf, save = flux_workflow("A calm teal gradient.", 770, 1365, seed=9)
    graph_refs_resolve(wf)
    assert wf["unet"]["inputs"]["unet_name"].startswith("flux3-image-klein")
    assert wf["clip"]["inputs"]["type"] == "flux3" and wf["vae"]["inputs"]["vae_name"] == "flux2-vae.safetensors"
    assert wf["sample"]["inputs"]["cfg"] == 1.0 and wf["sample"]["inputs"]["seed"] == 9
    assert wf["latent"]["inputs"]["width"] % 16 == 0 and wf["latent"]["inputs"]["height"] % 16 == 0
    assert wf["neg"]["inputs"]["text"] == ""  # distilled model: no negative prompt, so prompts describe, never negate






def test_art_screen_rejects_banners_and_lettering_but_not_ocr_noise(monkeypatch):
    from poster_agent.assets import generate as g
    from poster_agent.checks.ocr import OcrLine

    def lines(*items):
        return lambda img: [OcrLine(t, c, 0, 0, 10, 10) for t, c in items]

    noisy = Image.effect_noise((200, 200), 40).convert("RGB")
    monkeypatch.setattr(g, "read_lines", lines(("HEIUY GALEHE", 0.52), ("MILC", 0.55), ("?", 0.53), ("\u4e09", 0.5)))
    assert g.art_problem(noisy) is None  # hallucinations on fine patterns are not lettering
    monkeypatch.setattr(g, "read_lines", lines(("Image blocked by safety filter", 0.92)))
    assert "blocked" in g.art_problem(noisy)
    monkeypatch.setattr(g, "read_lines", lines(("Image clocknd by sgineer", 0.55)))  # a garbled banner at low confidence
    assert g.art_problem(noisy) is not None
    monkeypatch.setattr(g, "read_lines", lines(("SPRING SALE", 0.93)))  # real lettering, confidently read
    assert "SPRING SALE" in g.art_problem(noisy)
    monkeypatch.setattr(g, "read_lines", lines(("AB", 0.99), ("12", 0.99)))  # too short to be words
    assert g.art_problem(noisy) is None


def test_a_hard_straight_seam_rejects_a_background_but_not_a_soft_gradient():
    import numpy as np
    from PIL import Image
    from poster_agent.assets.generate import art_problem, seam_coverage
    seam = np.full((400, 300, 3), 150, dtype=np.uint8)
    seam[:140] -= 30  # a tonal step straight across the full width
    soft = np.tile(np.linspace(90, 200, 400, dtype=np.uint8)[:, None, None], (1, 300, 3))
    assert seam_coverage(Image.fromarray(seam))[0] >= 0.9
    assert "seam" in art_problem(Image.fromarray(seam), backdrop=True)
    assert art_problem(Image.fromarray(soft), backdrop=True) is None
    assert art_problem(Image.fromarray(seam)) is None  # only backdrops are held to this (objects have edges)
