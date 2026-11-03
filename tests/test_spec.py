import pytest
from pydantic import ValidationError

from poster_agent.spec import (AssetKind, AssetRef, Box, CanvasSpec, ComponentSpec, ElementKind, PlacedElement,
                               PosterSpec, find_unsourced_text)


def text_el(id_, text, **kw):
    return PlacedElement(id=id_, kind=ElementKind.text, role="title", box=Box(x=0, y=0, w=10, h=10), text=text, **kw)


def test_box_geometry():
    a, b = Box(x=0, y=0, w=10, h=10), Box(x=5, y=5, w=10, h=10)
    assert a.intersects(b) and a.intersection_area(b) == 25
    assert not a.intersects(Box(x=10, y=0, w=5, h=5))  # touching edges do not overlap
    assert a.contains(Box(x=1, y=1, w=8, h=8)) and not a.contains(b)
    assert a.inset(2) == Box(x=2, y=2, w=6, h=6) and a.expand(1).w == 12
    assert (a.cx, a.cy, a.right, a.bottom) == (5, 5, 10, 10)


def test_canvas_from_format():
    story = CanvasSpec.from_format("story")
    assert (story.width, story.height, story.export_scale) == (1080, 1920, 2.0)
    assert story.safe_box.x == pytest.approx(0.045 * 1080)
    a3 = CanvasSpec.from_format("A3")
    assert a3.kind == "print" and a3.bleed_pt == pytest.approx(3 * 72 / 25.4)
    with pytest.raises(ValueError):
        CanvasSpec.from_format("billboard")


def test_generated_logo_or_photo_is_rejected():
    # Guardrail 4
    for kind in (AssetKind.logo, AssetKind.photo, AssetKind.headshot, AssetKind.partner_logo):
        with pytest.raises(ValidationError):
            AssetRef(id="x", kind=kind, path="a.png", provenance="generated")
    AssetRef(id="bg", kind=AssetKind.generated_background, path="a.png", provenance="generated", seed=7)


def test_placed_element_requirements():
    with pytest.raises(ValidationError):
        PlacedElement(id="t", kind=ElementKind.text, role="title", box=Box(x=0, y=0, w=1, h=1))
    with pytest.raises(ValidationError):
        PlacedElement(id="i", kind=ElementKind.image, role="logo", box=Box(x=0, y=0, w=1, h=1))


def test_spec_reference_integrity_and_roundtrip():
    logo = AssetRef(id="logo", kind=AssetKind.logo, path="brands/kct/logos/quadrangle.png", provenance="brand")
    spec = PosterSpec(
        brief_id="b1", brand="kct", canvas=CanvasSpec.from_format("fourfive"),
        assets={"logo": logo},
        components=[ComponentSpec(id="c1", component="LogoBlock", asset_ids=["logo"])],
        placed=[PlacedElement(id="p1", kind=ElementKind.image, role="logo", asset_id="logo",
                              box=Box(x=10, y=10, w=100, h=100), locked=True)],
    )
    assert PosterSpec.model_validate_json(spec.model_dump_json()) == spec

    with pytest.raises(ValidationError):
        PosterSpec(brief_id="b", brand="kct", canvas=CanvasSpec.from_format("story"),
                   placed=[PlacedElement(id="p", kind=ElementKind.image, role="logo", asset_id="nope",
                                         box=Box(x=0, y=0, w=1, h=1))])
    with pytest.raises(ValidationError):
        PosterSpec(brief_id="b", brand="kct", canvas=CanvasSpec.from_format("story"),
                   placed=[text_el("dup", "A"), text_el("dup", "B")])


def test_verbatim_guard():
    sources = ["ANNUAL DAY 2026 & RE-LAUNCH", "Chief Guest: Dr. A. Kumar, Padma Shri Awardee"]
    ok = [
        text_el("a", "Annual Day 2026"),                      # part of a field, case-insensitive
        text_el("b", "ANNUAL DAY 2026\n& RE-LAUNCH"),          # layout-inserted wrap
        text_el("c", "Dr. A. Kumar\nPadma Shri Awardee"),  # one source fragment per line
    ]
    assert find_unsourced_text(ok, sources) == []
    invented = text_el("d", "Register now for a free lunch")
    assert find_unsourced_text(ok + [invented], sources) == [invented]
