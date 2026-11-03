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


def test_ideogram_workflow_is_wired_and_seeded():
    wf, save = ideogram_workflow({"high_level_description": "x"}, 900, 1130, seed=123, preset="turbo")
    graph_refs_resolve(wf)
    assert wf[save]["class_type"] == "SaveImage"
    assert wf["noise"]["inputs"]["noise_seed"] == 123
    assert wf["sigmas"]["inputs"]["steps"] == IDEOGRAM_PRESETS["turbo"]["steps"]
    assert wf["latent"]["inputs"]["width"] % 16 == 0 and wf["latent"]["inputs"]["height"] % 16 == 0
    assert json.loads(wf["pos"]["inputs"]["text"])["high_level_description"] == "x"  # structured JSON prompt


def test_qwen_workflow_is_wired():
    wf, save = qwen_workflow("a robot arm", 1000, 1000, seed=5)
    graph_refs_resolve(wf)
    assert wf["sample"]["inputs"]["seed"] == 5 and wf[save]["class_type"] == "SaveImage"
    assert "text" in wf["neg"]["inputs"]["text"]  # negative prompt keeps printed text out of the subject


def test_sizes_and_crops():
    for fmt in ("story", "fourfive", "square", "a3"):
        c = CanvasSpec.from_format(fmt)
        w, h = bg_size(c)
        assert w % 16 == 0 and h % 16 == 0 and 0.9e6 < w * h < 1.25e6
        assert abs(w / h - c.width / c.height) < 0.03
    assert round16(1010) == 1008
    img = cover_crop(Image.new("RGB", (1000, 500)), 1.0)
    assert img.size == (500, 500)
    assert cover_crop(Image.new("RGB", (500, 1000)), 1.0).size == (500, 500)


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


def test_calm_description_follows_the_layout():
    placed = [text_el(0, 225), text_el(900, 225)]  # exactly bands 0 and 4 of a 1000x1350 canvas
    cover = text_coverage_bands(placed, 1000, 1350)
    assert cover[0] == pytest.approx(1.0) and cover[4] == pytest.approx(1.0) and cover[1] == 0 and cover[5] == 0
    d = calm_description(placed, 1000, 1350)
    assert "very top" in d and "open space" in d and "keep" not in d.lower()  # a described scene, not an instruction
    lay = calm_layout(placed, 1000, 1350)
    assert lay.busy_ranges == [(167, 667), (833, 1000)]  # where visual interest may live, in 0-1000 bbox units
    assert "whole image" in calm_description([], 1000, 1350)
    full = [text_el(i * 225, 225) for i in range(6)]
    assert "calm, softly graded field" in calm_description(full, 1000, 1350)


def test_ideogram_prompt_structure_and_no_text_instructions():
    art = ArtDirection(description="Soft navy glass panels.", style="minimal", palette=["#1e2842"])
    p = ideogram_prompt(art, "Composition: keep the top calm.")
    assert p["high_level_description"].startswith("Soft navy glass panels.")
    assert "keep the top calm" in p["compositional_deconstruction"]["background"]
    assert p["style_description"]["color_palette"] == ["#1E2842"]  # the model wants uppercase hex
    flat = json.dumps(p).lower()
    assert "no text" not in flat and "typograph" not in flat  # Ideogram reacts badly to negative text incantations
    assert "isolated" in qwen_subject_prompt("a robot arm")


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
    assert any("set plate: true" in e for e in errs(p, setup))  # a logo on textured art needs a flat plate
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


def test_a_bad_image_already_in_the_cache_is_not_served(tmp_path):
    import hashlib

    from poster_agent.assets.generate import AssetGenerator, bg_size, derived_seed
    art, canvas = ArtDirection(description="Soft navy glass panels."), CanvasSpec.from_format("fourfive")
    calm = "calm"
    prompt = ideogram_prompt(art, calm)
    w, h = bg_size(canvas)
    seed = derived_seed("bg", json.dumps(prompt, sort_keys=True), f"{w}x{h}")
    wf, _ = ideogram_workflow(prompt, w, h, seed, "default", prefix="bg_bg")
    cache = tmp_path / "cache"
    cache.mkdir()
    poisoned = cache / (hashlib.sha1(json.dumps(wf, sort_keys=True).encode()).hexdigest() + ".png")
    poisoned.write_bytes(banner_png())  # e.g. left over from before screening existed
    comfy = Sequenced([noise_png(), noise_png()])
    ref, info = AssetGenerator(comfy, tmp_path / "out", cache).background("bg", art, canvas, calm, backend="ideogram")
    assert comfy.runs == 1 and ref.seed == seed + 1 and info.rejected  # the poisoned entry was refused; the next seed was drawn
    assert Image.open(ref.path if Path(ref.path).is_absolute() else ROOT / ref.path).size[0] > 0


def test_gives_up_with_every_rejection_reason_when_all_attempts_have_text(tmp_path):
    from poster_agent.assets.generate import AssetError, AssetGenerator
    gen = AssetGenerator(Sequenced([banner_png()] * 3), tmp_path / "out", tmp_path / "cache")
    with pytest.raises(AssetError, match="3 attempts.*blocked"):
        gen.background("bg", ArtDirection(description="x"), CanvasSpec.from_format("fourfive"), "calm")


# ---- the official Ideogram 4 caption schema (off-schema prompts risk a false "blocked by safety filter") -------------------

def test_built_captions_follow_the_ideogram_schema():
    art = ArtDirection(description="Translucent navy glass panels with a faint dotted grid", style="clean minimal geometric",
                       lighting="soft diffused morning light", medium="digital illustration", palette=["#1e2842", "#85cebd"])
    for placed in ([], [text_el(0, 225), text_el(900, 225)], [text_el(i * 225, 225) for i in range(6)], [text_el(300, 600)]):
        cap = ideogram_prompt(art, calm_layout(placed, 1000, 1350))
        assert validate_ideogram_caption(cap) == [], (placed, validate_ideogram_caption(cap))
    cap = ideogram_prompt(art, calm_layout([text_el(0, 225)], 1000, 1350))
    assert list(cap) == ["high_level_description", "style_description", "compositional_deconstruction"]
    assert list(cap["style_description"]) == ["aesthetics", "lighting", "medium", "art_style", "color_palette"]  # non-photo order
    assert cap["compositional_deconstruction"]["elements"][0]["bbox"] == [167, 0, 1000, 1000]  # [y_min, x_min, y_max, x_max]
    photo = ideogram_prompt(ArtDirection(description="A quiet campus lawn at dawn", medium="photograph"), "calm")
    assert list(photo["style_description"]) == ["aesthetics", "lighting", "photo", "medium"] and validate_ideogram_caption(photo) == []


def test_validator_catches_every_off_schema_mistake_we_used_to_make():
    good = ideogram_prompt(ArtDirection(description="x y z", palette=["#1E2842"]), "calm")
    bad = copy.deepcopy(good)
    del bad["style_description"]["art_style"]                       # neither photo nor art_style
    assert any("exactly one of photo / art_style" in e for e in validate_ideogram_caption(bad))
    bad = copy.deepcopy(good)
    bad["style_description"] = {"aesthetics": "a", "medium": "illustration", "lighting": "l", "art_style": "s"}  # wrong key order
    assert any("required order" in e for e in validate_ideogram_caption(bad))
    bad = copy.deepcopy(good)
    bad["style_description"]["color_palette"] = ["#1e2842"]         # lowercase hex
    assert any("uppercase" in e for e in validate_ideogram_caption(bad))
    bad = copy.deepcopy(good)
    bad["compositional_deconstruction"]["elements"][0]["bbox"] = [900, 0, 100, 1000]  # min > max
    assert any("bbox" in e for e in validate_ideogram_caption(bad))
    bad = copy.deepcopy(good)
    del bad["compositional_deconstruction"]
    assert validate_ideogram_caption(bad)


def test_caption_is_serialised_compactly_with_literal_unicode():
    cap = ideogram_prompt(ArtDirection(description="Kumaraguru கல்லூரி campus lawn"), "calm")
    text = caption_text(cap)
    assert text == json.dumps(json.loads(text), separators=(",", ":"), ensure_ascii=False)  # compact: no indent, no spaces
    assert "கல்லூரி" in text and "\\u" not in text  # literal unicode, not \uXXXX escapes
    wf, _ = ideogram_workflow(cap, 900, 1130, seed=1)
    assert wf["pos"]["inputs"]["text"] == text


def test_off_schema_captions_are_never_sent_to_the_gpu(tmp_path, monkeypatch):
    from poster_agent.assets import generate as g
    monkeypatch.setattr(g, "ideogram_prompt", lambda art, calm: {"high_level_description": "plain text style"})
    comfy = Sequenced([noise_png()])
    gen = g.AssetGenerator(comfy, tmp_path / "out", tmp_path / "cache")
    with pytest.raises(g.AssetError, match="Ideogram schema"):
        gen.background("bg", ArtDirection(description="x"), CanvasSpec.from_format("fourfive"), "calm", backend="ideogram")
    assert comfy.runs == 0


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


def test_flux_prompt_describes_the_scene_palette_and_where_interest_sits():
    from poster_agent.assets.prompts import color_words, flux_prompt, focus_phrase
    assert color_words(["#1e2842", "#85cebd", "#dde3ed"]) == "navy, mint and pale grey-blue"
    assert focus_phrase([(667, 1000)]) == "in the lower third of the frame" and focus_phrase([(0, 1000)]) == "across most of the frame"
    art = ArtDirection(description="Translucent glass panels with a faint dotted grid", style="clean minimal", lighting="soft morning light",
                       palette=["#1e2842", "#85cebd"])
    p = flux_prompt(art, calm_layout([text_el(0, 450)], 1000, 1350))
    assert p.startswith("Translucent glass panels") and "The colour palette is navy and mint." in p
    assert "open space fills the very top, upper" in p and "interest sits in the lower half of the frame" in p
    low = p.lower()
    assert " no " not in low and "don't" not in low and "without" not in low  # nothing negated: the model has no negative prompt


def test_backgrounds_default_to_flux_and_ideogram_stays_selectable(tmp_path):
    from poster_agent.assets.generate import AssetGenerator
    comfy = Sequenced([noise_png(), noise_png()])
    gen = AssetGenerator(comfy, tmp_path / "out", tmp_path / "cache")
    art, canvas = ArtDirection(description="Soft navy glass panels."), CanvasSpec.from_format("fourfive")
    _, info = gen.background("a", art, canvas, "calm")
    assert info.model == "flux3_klein" and "Soft navy glass panels." in info.prompt
    _, info2 = gen.background("b", art, canvas, "calm", backend="ideogram")
    assert info2.model == "ideogram4" and json.loads(info2.prompt)["high_level_description"]
    with pytest.raises(ValueError):
        gen.background("c", art, canvas, "calm", backend="midjourney")


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
