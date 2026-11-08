"""Follow-up test: does Ideogram 4 keep poster text clean when every text element has a box, there are only a few SHORT lines, and the
medium is photo / 3D / illustration (not graphic_design)?  The first poster test had none of these and produced garbled text and blocked images.

Usage: PYTHONPATH=. python scripts/ideogram_box_test.py [--seeds 2]   -> eval/reference/box_<name>_<seed>.png + sheet_box_<name>.png"""
from __future__ import annotations

import argparse
import sys
import time

from PIL import Image

from poster_agent.assets.comfy import ComfyClient
from poster_agent.assets.prompts import validate_ideogram_caption
from poster_agent.assets.workflows import ideogram_workflow
from poster_agent.spec.brand import ROOT, load_brand

OUT = ROOT / "eval" / "reference"


def captions(pal: list[str]) -> dict[str, tuple[dict, int, int]]:
    """name -> (caption, width, height). Boxes are [y_min, x_min, y_max, x_max] on a 0-1000 grid."""
    return {
        "robotics_day": ({
            "high_level_description": "A futuristic event poster: an industrial robot arm in a deep navy technology hall with a bold headline above it.",
            "style_description": {"aesthetics": "futuristic, professional, high contrast", "lighting": "cool blue rim light with soft glow",
                                  "medium": "3d_render", "art_style": "polished product render, deep navy atmosphere", "color_palette": pal},
            "compositional_deconstruction": {
                "background": "A deep navy technology hall with soft glowing circuit lines and a calm dark upper area.",
                "elements": [
                    {"type": "text", "bbox": [70, 60, 250, 940], "text": "ROBOTICS DAY 2026", "desc": "Huge bold white sans-serif headline"},
                    {"type": "text", "bbox": [260, 120, 320, 880], "text": "Where machines meet imagination",
                     "desc": "Light mint sans-serif tagline"},
                    {"type": "obj", "bbox": [360, 120, 930, 900], "desc": "A sleek silver industrial robot arm reaching upward, glowing blue joints"},
                    {"type": "text", "bbox": [940, 60, 985, 940], "text": "FRIDAY, 14 AUGUST 2026", "desc": "Small bold white caps"}]}}, 720, 1280),
        "placement_drive": ({
            "high_level_description": "A clean corporate recruitment poster with a white upper area, a big navy and mint headline and a navy lower panel with a curved edge.",
            "style_description": {"aesthetics": "clean, formal, trustworthy, flat vector shapes", "lighting": "flat, even",
                                  "medium": "illustration", "art_style": "flat corporate vector design", "color_palette": pal},
            "compositional_deconstruction": {
                "background": "White upper two thirds, and a deep navy panel across the bottom with a gently curved top edge.",
                "elements": [
                    {"type": "text", "bbox": [120, 70, 420, 930], "text": "CAMPUS PLACEMENT DRIVE 2027",
                     "desc": "Very large bold navy sans-serif headline, last words in mint"},
                    {"type": "text", "bbox": [450, 100, 520, 900], "text": "Your career starts here", "desc": "Medium navy sans-serif tagline"},
                    {"type": "obj", "bbox": [560, 0, 1000, 1000], "desc": "A deep navy panel with a curved top edge"},
                    {"type": "text", "bbox": [700, 80, 790, 920], "text": "22 FEBRUARY 2027 | PLACEMENT HALL",
                     "desc": "Bold white caps on the navy panel"}]}}, 896, 1120),
        "sports_meet": ({
            "high_level_description": "A sports event poster over a sunlit athletics stadium with a strong headline across the top.",
            "style_description": {"aesthetics": "energetic, bright, professional", "lighting": "warm low evening sun with long shadows",
                                  "photo": "wide shot, shallow depth of field", "medium": "photograph", "color_palette": pal},
            "compositional_deconstruction": {
                "background": "A running track and green field in a stadium at golden hour, with calm open space at the top.",
                "elements": [
                    {"type": "text", "bbox": [60, 50, 300, 950], "text": "INTER-DEPARTMENT SPORTS MEET",
                     "desc": "Large bold condensed navy sans-serif headline in two lines"},
                    {"type": "text", "bbox": [320, 200, 390, 800], "text": "FROM 18 JANUARY 2027", "desc": "Bold white caps on a navy strip"},
                    {"type": "obj", "bbox": [450, 0, 1000, 1000], "desc": "Athletics track lanes curving across a green field"},
                    {"type": "text", "bbox": [900, 200, 960, 800], "text": "Play hard. Play fair.", "desc": "Bold white italic tagline"}]}}, 1024, 1024),
    }


def no_text(cap: dict, name: str) -> dict:
    """Same layout, but every text element becomes an empty calm area at the same box (no lettering requested), and nobody is pictured."""
    import copy
    c = copy.deepcopy(cap)
    els = []
    for e in c["compositional_deconstruction"]["elements"]:
        if e["type"] == "text":
            els.append({"type": "obj", "bbox": e["bbox"], "desc": "A calm, smooth, softly graded open area of even tone, kept free for a headline"})
        else:
            if name == "sports_meet":
                e = {**e, "desc": "Athletics track lanes curving across a green field in an empty stadium"}
            els.append(e)
    c["compositional_deconstruction"]["elements"] = els
    return c


def report(name: str, cap: dict, w: int, h: int, seed: int, path) -> str:
    """Measured, not eyeballed: lettering found by OCR, and fine detail under the reserved boxes against the whole picture."""
    from poster_agent.analysis import text_detail
    from poster_agent.assets.generate import art_problem
    from poster_agent.spec import Box
    img = Image.open(path).convert("RGB")
    boxes = [Box(x=e["bbox"][1], y=e["bbox"][0], w=e["bbox"][3] - e["bbox"][1], h=e["bbox"][2] - e["bbox"][0])
             for e in cap["compositional_deconstruction"]["elements"] if e["desc"].startswith("A calm")]
    under = text_detail(img, boxes, 1000, 1000)
    whole = text_detail(img, [Box(x=0, y=0, w=1000, h=1000)], 1000, 1000)
    prob = art_problem(img)
    return f"{name} seed {seed}: detail under reserved boxes {under:.4f} vs whole picture {whole:.4f} (ratio {under / max(whole, 1e-6):.2f}); lettering: {prob or 'none found'}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--no-text", action="store_true", help="reserve empty areas instead of asking for text")
    args = ap.parse_args()
    brand = load_brand("kct")
    pal = [brand.palette[k].upper() for k in ("primary", "accent", "surface") if k in brand.palette]
    OUT.mkdir(parents=True, exist_ok=True)
    comfy = ComfyClient()
    for name, (cap, w, h) in captions(pal).items():
        if args.no_text:
            cap = no_text(cap, name)
        bad = validate_ideogram_caption(cap)
        if bad:
            print(f"{name}: caption off-schema, not sent: {bad}", flush=True)
            continue
        tiles = []
        for seed in range(1, args.seeds + 1):
            t = time.time()
            try:
                wf, node = ideogram_workflow(cap, w, h, seed, prefix=f"box_{name}")
                data = comfy.run(wf, node)
            except Exception as e:  # noqa: BLE001
                print(f"{name} seed {seed}: FAILED {str(e)[:150]}", flush=True)
                continue
            p = OUT / (f"empty_{name}_{seed}.png" if args.no_text else f"box_{name}_{seed}.png")
            p.write_bytes(data)
            tiles.append(Image.open(p).convert("RGB"))
            print(f"{name} seed {seed}: {time.time() - t:.0f}s -> {p.relative_to(ROOT)}", flush=True)
            if args.no_text:
                print("   " + report(name, cap, w, h, seed, p), flush=True)
        if tiles:
            th = 900
            tiles = [t.resize((int(t.width * th / t.height), th)) for t in tiles]
            sheet = Image.new("RGB", (sum(t.width for t in tiles) + 12 * (len(tiles) - 1), th), "white")
            x = 0
            for t in tiles:
                sheet.paste(t, (x, 0))
                x += t.width + 12
            sheet.save(OUT / (f"sheet_empty_{name}.png" if args.no_text else f"sheet_box_{name}.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
