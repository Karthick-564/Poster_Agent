"""Person photo framing: crop only, head found from the matte; tight photos are reported as limited, never invented."""
import numpy as np
from PIL import Image, ImageDraw

from poster_agent.assets.person import PRESETS, frame_person, head_from_mask, plan_crop


def person(w=600, h=900, head_r=70, shoulders=True):
    """Synthetic bust: a round head on a neck over wide shoulders, as a boolean matte and a photo."""
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    cx, top = w // 2, 120
    d.ellipse((cx - head_r, top, cx + head_r, top + 2 * head_r + 20), fill=255)  # head
    d.rectangle((cx - 28, top + 2 * head_r, cx + 28, top + 2 * head_r + 70), fill=255)  # neck
    if shoulders:
        d.polygon([(cx - 28, top + 2 * head_r + 60), (cx + 28, top + 2 * head_r + 60), (cx + 230, top + 2 * head_r + 200),
                   (cx + 230, h), (cx - 230, h), (cx - 230, top + 2 * head_r + 200)], fill=255)
    mask = np.array(m) > 0
    photo = Image.new("RGB", (w, h), (200, 205, 210))
    photo.paste((120, 90, 70), mask=m)
    return photo, mask


def test_head_is_found_above_the_neck_not_the_whole_body():
    _, mask = person()
    head = head_from_mask(mask)
    assert head is not None
    assert 120 <= head.y <= 125 and 130 <= head.w <= 150 and 140 <= head.h <= 190  # about the head ellipse, neck excluded


def test_head_and_shoulders_crop_ends_below_the_chin_and_is_centred():
    photo, mask = person()
    out, fr = frame_person(photo, "head_and_shoulders", mask=mask)
    l, t, r, b = fr.box
    head = fr.head
    assert t <= head.y and b > head.y + head.h + 0.7 * head.h  # includes the shoulders area
    assert b < photo.height  # but is not the whole body
    assert abs((l + r) / 2 - 300) < 5 and out.size == (r - l, b - t)
    assert abs(out.size[0] / out.size[1] - PRESETS["head_and_shoulders"][3]) < 0.03


def test_the_cropped_pixels_are_the_original_pixels():
    photo, mask = person()
    out, fr = frame_person(photo, "head", mask=mask)
    l, t, r, b = fr.box
    assert np.array_equal(np.asarray(out), np.asarray(photo.crop((l, t, r, b))))  # a crop, nothing repainted


def test_a_tight_photo_cannot_become_half_body_and_says_so():
    photo, mask = person(h=420, shoulders=False)  # head and neck only, little below
    _, fr = frame_person(photo, "half_body", mask=mask)
    assert fr.limited and "does not extend" in fr.note
    l, t, r, b = fr.box
    assert l >= 0 and t >= 0 and r <= photo.width and b <= photo.height


def test_no_person_is_reported_not_faked():
    blank = Image.new("RGB", (400, 400), "white")
    out, fr = frame_person(blank, "head", mask=np.zeros((400, 400), dtype=bool))
    assert fr.method == "none" and fr.limited and "no person found" in fr.note and out.size == (400, 400)


def test_plan_crop_keeps_the_box_inside_the_photo():
    from poster_agent.assets.person import Head
    box, limited = plan_crop((500, 500), Head(x=350, y=20, w=120, h=140), "head_and_shoulders")
    assert box[0] >= 0 and box[2] <= 500 and box[1] >= 0 and box[3] <= 500


def test_a_supplied_person_photo_becomes_a_matted_copy_and_a_failed_matte_leaves_the_photo_alone():
    from poster_agent.assets.cutouts import apply_cutouts
    from poster_agent.layout.solver import solve_layout
    from poster_agent.spec import AssetKind, AssetRef, CanvasSpec, ComponentSpec, PosterSpec
    from poster_agent.spec.brand import ROOT, load_brand
    d = ROOT / "runs/_tests"
    d.mkdir(parents=True, exist_ok=True)
    photo, _ = person()
    photo.save(d / "ppl.png")

    def fresh():
        asset = AssetRef(id="ppl", kind=AssetKind.headshot, path="runs/_tests/ppl.png", provenance="supplied")
        comp = ComponentSpec(id="a", component="BottomCutout", anchor="bottom", content={"asset": "ppl", "align": "right"})
        sp = PosterSpec(brief_id="t", brand="kct", canvas=CanvasSpec.from_format("story"), components=[comp], assets={"ppl": asset})
        sp.placed = solve_layout(sp, load_brand("kct")).placed
        return sp

    def half(im, pad_pct=0.0):  # a stand-in matte: the left half opaque
        rgba = im.convert("RGBA"); a = Image.new("L", im.size, 0); a.paste(255, (0, 0, im.width // 2, im.height)); rgba.putalpha(a); return rgba

    spec = fresh()
    notes = apply_cutouts(spec, d / "cut", matte=half)
    img = next(e for e in spec.placed if e.id == "a.img")
    assert "cutout" in img.asset_id and spec.assets[img.asset_id].treatment["cutout"] and any("background" in n for n in notes)
    assert spec.assets["ppl"].path == "runs/_tests/ppl.png"  # the supplied file itself is untouched

    spec = fresh()
    notes = apply_cutouts(spec, d / "cut", matte=lambda im, pad_pct=0.0: im.convert("RGBA"))  # a matte that kept everything failed
    img = next(e for e in spec.placed if e.id == "a.img")
    assert img.asset_id == "ppl" and any("could not be separated" in n for n in notes)
