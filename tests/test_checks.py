"""Planted-failure tests for the pixel checks, on small synthetic renders (no VectorCraft needed)."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from poster_agent.checks import run_checks
from poster_agent.checks import ocr, pixels
from poster_agent.compile.compiler import CompileResult
from poster_agent.compile.vc_client import cli_path
from poster_agent.spec import (AssetKind, AssetRef, Box, CanvasSpec, ElementKind, IssueCategory, MandatoryItem, PlacedElement,
                               PosterSpec)
from poster_agent.spec.brand import ROOT, load_brand

KCT = load_brand("kct")
needs_vc = pytest.mark.skipif(not cli_path().exists(), reason="vectorcraft-cli not installed")
FONT = str(ROOT / "brands/fonts/Montserrat-Bold.ttf")
W = H = 1000


def canvas() -> CanvasSpec:
    return CanvasSpec(format="square", width=W, height=H, export_scale=1.0)


def draw_text(img: Image.Image, xy, text, size=60, fill=(30, 40, 66)):
    ImageDraw.Draw(img).text(xy, text, font=ImageFont.truetype(FONT, size), fill=fill)


def text_el(id_, text, box, color="#1e2842", size=60) -> PlacedElement:
    return PlacedElement(id=id_, kind=ElementKind.text, role="title", box=Box(x=box[0], y=box[1], w=box[2], h=box[3]), text=text,
                         font="Montserrat", style="Bold", size=size, color=color, z=10)


def make(tmp_path: Path, img: Image.Image, placed, mandatory=None, assets=None, scale=1.0, overflow=None, warnings=None):
    png = tmp_path / "p.png"
    img.save(png)
    pdf, vcf = tmp_path / "p.pdf", tmp_path / "p.vectorcraft"
    pdf.write_bytes(b"x")
    vcf.write_bytes(b"x")
    spec = PosterSpec(brief_id="t", brand="kct", canvas=canvas(), placed=placed, mandatory=mandatory or [], assets=assets or {})
    compiled = CompileResult(png=png, pdf=pdf, vectorcraft=vcf, bounds={e.id: e.box for e in placed}, overflow=overflow or [],
                             warnings=warnings or [])
    return spec, compiled


def cats(rep):
    return {i.category for i in rep.issues if i.severity == "error"}


# ---- pixel primitives -------------------------------------------------------------------------------------------------

def test_contrast_measurement():
    img = Image.new("RGB", (200, 100), "white")
    draw_text(img, (10, 10), "Hi", size=60, fill=(255, 255, 255))  # white text on white: invisible
    low = pixels.text_contrast(img, Box(x=0, y=0, w=200, h=100), 1.0, (255, 255, 255))
    assert low["ratio"] is None or low["ratio"] < 1.1
    img2 = Image.new("RGB", (200, 100), (30, 40, 66))
    good = pixels.text_contrast(img2, Box(x=0, y=0, w=200, h=100), 1.0, (255, 255, 255))
    assert good["ratio"] > 12
    grad = Image.fromarray(np.tile(np.linspace(255, 0, 200, dtype=np.uint8), (100, 1))).convert("RGB")  # white -> black ground
    worst = pixels.text_contrast(grad, Box(x=0, y=0, w=200, h=100), 1.0, (255, 255, 255))
    assert worst["ratio"] < 3  # judged at its worst percentile, not the average


def test_ocr_matching_ignores_spacing_and_punctuation():
    assert ocr.match_score("KUMARAGURU CENTRE FOR INDUSTRIAL", "KUMARAGURUCENTREFORINDUSTRIAL") == 100
    assert ocr.match_score("MONDAY, 27 JULY 2026 | 10:30 AM", "MONDAY, 27 JULY 2026 I 10:30 AM") > 90
    assert ocr.match_score("RE-LAUNCH", "something else entirely") < 60  # below the 60 failure threshold
    assert not ocr.verifiable("குமரகுரு") and ocr.verifiable("Annual Day")


# ---- runner without OCR ---------------------------------------------------------------------------------------------

def test_clean_text_poster_passes_geometry_and_contrast(tmp_path):
    img = Image.new("RGB", (W, H), "white")
    draw_text(img, (100, 100), "ANNUAL DAY")
    spec, compiled = make(tmp_path, img, [text_el("t", "ANNUAL DAY", (100, 100, 600, 80))])
    rep = run_checks(spec, KCT, compiled, use_ocr=False, use_emptiness=False)
    # this fixture deliberately has no logo, so only the logo-presence rule may fail
    assert {i.category for i in rep.issues if i.severity == "error"} == {IssueCategory.logo_violation}, [i.evidence for i in rep.issues]
    assert all("no institution logo" in i.evidence for i in rep.issues if i.severity == "error")


def test_invisible_text_is_flagged_by_contrast(tmp_path):
    img = Image.new("RGB", (W, H), (30, 40, 66))
    draw_text(img, (100, 100), "ANNUAL DAY", fill=(36, 46, 72))  # navy on navy
    spec, compiled = make(tmp_path, img, [text_el("t", "ANNUAL DAY", (100, 100, 600, 80))])
    assert IssueCategory.low_contrast in cats(run_checks(spec, KCT, compiled, use_ocr=False, use_emptiness=False))


def test_overflow_font_fallback_size_and_margin_are_reported(tmp_path):
    img = Image.new("RGB", (W // 2, H // 2), "white")  # wrong size
    spec, compiled = make(tmp_path, img, [text_el("edge", "X", (2, 2, 100, 40))], overflow=["edge"],
                          warnings=[".png: fonts that aren't available were written in the fallback font"])
    c = cats(run_checks(spec, KCT, compiled, use_ocr=False, use_emptiness=False))
    assert {IssueCategory.output_spec, IssueCategory.fit, IssueCategory.margin_violation} <= c


def test_text_overlap_on_real_bounds(tmp_path):
    img = Image.new("RGB", (W, H), "white")
    spec, compiled = make(tmp_path, img, [text_el("a", "AAA", (100, 100, 400, 80)), text_el("b", "BBB", (150, 120, 400, 80))])
    assert IssueCategory.text_overlap in cats(run_checks(spec, KCT, compiled, use_ocr=False, use_emptiness=False))


# ---- logo rules ----------------------------------------------------------------------------------------------------

def logo_case(tmp_path, *, size=130, xy=(100, 100), cover=False, gradient=False, intruder=False):
    art = Image.open(ROOT / KCT.logo("quadrangle").path).convert("RGB")
    img = Image.new("RGB", (W, H), "white")
    if gradient:
        img.paste(Image.fromarray(np.tile(np.linspace(255, 120, 300, dtype=np.uint8), (300, 1))).convert("RGB"), (xy[0] - 80, xy[1] - 80))
    img.paste(art.resize((size, size)), xy)
    if cover:
        ImageDraw.Draw(img).rectangle((xy[0], xy[1], xy[0] + size // 2, xy[1] + size), fill="white")
    placed = [PlacedElement(id="logo", kind=ElementKind.image, role="logo", asset_id="brand_logo_quadrangle",
                            box=Box(x=xy[0], y=xy[1], w=size, h=size), locked=True)]
    if intruder:
        placed.append(text_el("t", "HELLO", (xy[0] + size + 8, xy[1], 200, 40), size=30))
    assets = {"brand_logo_quadrangle": AssetRef(id="brand_logo_quadrangle", kind=AssetKind.logo, path=KCT.logo("quadrangle").path,
                                                provenance="brand")}
    return make(tmp_path, img, placed, assets=assets)


def logo_problems(rep) -> str:
    return " | ".join(i.evidence for i in rep.issues if i.category == IssueCategory.logo_violation)


def test_correct_logo_passes(tmp_path):
    spec, compiled = logo_case(tmp_path)
    assert logo_problems(run_checks(spec, KCT, compiled, use_ocr=False, use_emptiness=False)) == ""


def test_logo_covered_is_detected(tmp_path):
    spec, compiled = logo_case(tmp_path, cover=True)
    assert "does not match the artwork" in logo_problems(run_checks(spec, KCT, compiled, use_ocr=False, use_emptiness=False))


def test_logo_too_small_edge_and_exclusion_zone(tmp_path):
    spec, compiled = logo_case(tmp_path, size=60)  # 6% < 10% minimum
    assert "below the 10% minimum" in logo_problems(run_checks(spec, KCT, compiled, use_ocr=False, use_emptiness=False))
    spec, compiled = logo_case(tmp_path, xy=(12, 12))
    assert "canvas edge" in logo_problems(run_checks(spec, KCT, compiled, use_ocr=False, use_emptiness=False))
    spec, compiled = logo_case(tmp_path, intruder=True)
    assert "exclusion zone" in logo_problems(run_checks(spec, KCT, compiled, use_ocr=False, use_emptiness=False))


def _brand_with_solid_field(value: bool):
    b = KCT.model_copy(deep=True)
    for v in b.logos.values():
        v.solid_field = value
    return b


def test_logo_on_gradient_is_detected_unless_the_logo_brings_its_own_field(tmp_path):
    spec, compiled = logo_case(tmp_path, gradient=True)
    open_logo = _brand_with_solid_field(False)  # artwork without a field of its own needs a flat ground
    assert "not on a flat field" in logo_problems(run_checks(spec, open_logo, compiled, use_ocr=False, use_emptiness=False))
    own_field = _brand_with_solid_field(True)  # a navy square carries its field: any ground is fine
    assert "not on a flat field" not in logo_problems(run_checks(spec, own_field, compiled, use_ocr=False, use_emptiness=False))


# ---- OCR on the render ----------------------------------------------------------------------------------------------

def test_ocr_finds_missing_mandatory_and_stray_text(tmp_path):
    img = Image.new("RGB", (W, H), "white")
    draw_text(img, (100, 100), "ANNUAL DAY 2026", size=70)
    draw_text(img, (100, 600), "SURPRISE BANNER", size=70)  # baked-in text nobody placed (e.g. inside generated art)
    placed = [text_el("title", "ANNUAL DAY 2026", (100, 90, 800, 100), size=70),
              text_el("venue", "KUMARAGURU CAMPUS", (100, 300, 800, 100), size=70)]  # declared but never drawn
    mandatory = [MandatoryItem(id="title", label="title", text="ANNUAL DAY 2026"),
                 MandatoryItem(id="venue", label="venue", text="KUMARAGURU CAMPUS")]
    spec, compiled = make(tmp_path, img, placed, mandatory)
    rep = run_checks(spec, KCT, compiled, use_emptiness=False)
    errs = [i for i in rep.issues if i.severity == "error"]
    assert any(i.category == IssueCategory.mandatory_missing and "venue" in i.evidence for i in errs)
    assert any(i.category == IssueCategory.stray_text and "SURPRISE" in i.evidence.upper() for i in errs)
    assert not any(i.category == IssueCategory.mandatory_missing and i.element_ids == ["title"] for i in errs)  # drawn text is fine
    names = {r.name: r.passed for r in rep.results}
    assert names["mandatory_text"] is False and names["stray_text"] is False and names["ocr_elements"] is False


def test_clean_ocr_case_passes_and_non_latin_is_unverifiable_not_failed(tmp_path):
    img = Image.new("RGB", (W, H), "white")
    draw_text(img, (100, 100), "ANNUAL DAY 2026", size=70)
    placed = [text_el("title", "ANNUAL DAY 2026", (100, 90, 800, 100), size=70),
              text_el("tamil", "குமரகுரு", (100, 400, 400, 80), size=40)]
    spec, compiled = make(tmp_path, img, placed, [MandatoryItem(id="title", label="title", text="ANNUAL DAY 2026"),
                                                  MandatoryItem(id="ta", label="ta", text="குமரகுரு")])
    rep = run_checks(spec, KCT, compiled, use_emptiness=False)
    ocr_res = next(r for r in rep.results if r.name == "ocr_elements")
    assert ocr_res.measured["unverifiable"] == ["tamil"]
    assert not any(i.category == IssueCategory.mandatory_missing for i in rep.issues if i.severity == "error")


# ---- regressions for defects the checks found in our own components -----------------------------------------------------





def test_flat_field_test_ignores_the_logos_own_edge_pixels():
    img = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(img).rectangle((50, 50, 149, 149), fill=(30, 40, 66))  # logo-like block with a hard edge
    strips = pixels.ring_strips(Box(x=50, y=50, w=100, h=100), 20, 200, 200)
    solid = pixels.field_solidity(img, strips, 1.0)
    assert solid["max_std"] < 2 and solid["mean_lum"] > 250  # the surrounding field is flat white




def test_ocr_reads_tight_thin_crops():
    """Regression: wide crops only ~70px tall used to return '' (no margin for the detector), flagging visible text as missing."""
    img = Image.new("RGB", (2000, 400), "white")
    draw_text(img, (40, 160), "CELEBRATING RESEARCH, INNOVATION & NATIONAL DEVELOPMENT", size=44, fill=(70, 80, 100))
    box = (0, 150, 2000, 150 + 72)  # tight around the text line
    got = ocr.read_crop(img, box)
    assert ocr.match_score("CELEBRATING RESEARCH, INNOVATION & NATIONAL DEVELOPMENT", got) >= 90, got




def test_a_dropped_mandatory_line_is_retried_alone_and_a_truly_missing_one_still_fails(tmp_path, monkeypatch):
    from poster_agent.checks import ocr, run_checks
    from poster_agent.spec import MandatoryItem
    block = "FIRST LINE HERE\nSECOND LINE HERE\nTHIRD LINE HERE"
    img = Image.new("RGB", (1000, 1000), "white")
    spec, compiled = make(tmp_path, img, [text_el("blk", block, (100, 100, 700, 200), size=40)],
                          mandatory=[MandatoryItem(id="m", label="third", text="THIRD LINE HERE")])
    reads = []

    def fake_read(image, box, *a, **k):
        reads.append(box)
        return "FIRST LINE HERE SECOND LINE HERE" if len(reads) == 1 else "THIRD LINE HERE"  # the block read drops line 3; the line read finds it
    monkeypatch.setattr(ocr, "read_crop", fake_read)
    rep = run_checks(spec, KCT, compiled, use_ocr=True, use_emptiness=False)
    mand = next(r for r in rep.results if r.name == "mandatory_text")
    assert mand.passed and len(reads) >= 2  # the missing line was cropped out and read on its own
    reads.clear()
    monkeypatch.setattr(ocr, "read_crop", lambda *a, **k: "FIRST LINE HERE SECOND LINE HERE")  # nothing ever reads line 3
    assert not next(r for r in run_checks(spec, KCT, compiled, use_ocr=True, use_emptiness=False).results if r.name == "mandatory_text").passed
