"""Experiment: how good is an image model at designing a WHOLE poster (typography included) from a brief?

The result is a design REFERENCE only: its text is never trusted (image models garble letters and invent logos). It tells us whether
'generate a reference design, then rebuild it with real text, real logo and editable layers' is worth building.

Usage:  PYTHONPATH=. python scripts/reference_poster.py [brief names ...] [--seeds 1]
Writes eval/reference/<brief>_<model>_<seed>.png and a contact sheet eval/reference/sheet_<brief>.png. Needs the GPU box (COMFYUI_URL)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

from poster_agent.assets.comfy import ComfyClient
from poster_agent.assets.generate import bg_size
from poster_agent.assets.prompts import color_words, validate_ideogram_caption
from poster_agent.assets.workflows import flux_workflow, ideogram_workflow, qwen_workflow
from poster_agent.ingest.brief import Brief, load_brief
from poster_agent.spec import CanvasSpec
from poster_agent.spec.brand import ROOT, load_brand

OUT = ROOT / "eval" / "reference"


def poster_prompt(b: Brief, palette_hexes: list[str]) -> str:
    """A designer's brief to an image model: the design intent plus the exact words to set (quoted, in reading order)."""
    words = [("headline", b.event_title), ("subheading", b.subtitle), ("tagline", b.tagline), ("organiser", b.organizer)]
    details = " | ".join(x for x in (b.date, b.time, b.venue) if x)
    lines = [f'the {k} reads "{v}"' for k, v in words if v]
    if details:
        lines.append(f'the details line reads "{details}"')
    lines += [f'a line reads "{t}"' for t in (b.theme_lines + b.extra_lines)[:4]]
    if b.cta:
        lines.append(f'the call to action reads "{b.cta}"')
    colours = color_words(palette_hexes)
    notes = f" Mood: {b.style_notes}" if b.style_notes else ""
    return ("A professionally designed institutional event poster for a university, portrait format, with a clear typographic hierarchy: "
            "one dominant headline, supporting lines, generous structure and one strong visual idea that fits the event. "
            f"The colour palette is {colours}. A clean blank space is kept at the top for a logo. Typography: " + "; ".join(lines) + "." + notes)


def ideogram_caption(b: Brief, palette_hexes: list[str]) -> dict:
    """The same design intent as `poster_prompt`, in Ideogram 4's strict caption format (see assets/prompts.py). Each printed line is a text
    element with its exact words; no boxes are given, so the model chooses the layout."""
    elements: list[dict] = []
    roles = [(b.event_title, "the dominant headline, very large bold sans-serif"), (b.subtitle, "supporting heading, medium weight"),
             (b.tagline, "tagline, light weight"), (b.organizer, "small caps organiser line")]
    details = " | ".join(x for x in (b.date, b.time, b.venue) if x)
    roles.append((details, "event details line, bold, small"))
    roles += [(t, "supporting line, regular weight") for t in (b.theme_lines + b.extra_lines)[:4]]
    roles.append((b.cta, "call to action, bold"))
    for text, desc in roles:
        if text:
            elements.append({"type": "text", "text": text, "desc": desc})
    palette = [c.upper() for c in palette_hexes][:5]
    return {
        "high_level_description": (f"A professionally designed institutional event poster for {b.event_title}, portrait format, with a clear "
                                   "typographic hierarchy, generous structure and one strong visual idea that fits the event, with an empty "
                                   "space at the top for a logo." + (f" Mood: {b.style_notes}" if b.style_notes else "")),
        "style_description": {"aesthetics": "clean, modern, institutional, bold typography", "lighting": "soft, even, professional",
                              "medium": "graphic_design", "art_style": "contemporary corporate poster design",
                              "color_palette": [c.upper() for c in palette_hexes][:16]},
        "compositional_deconstruction": {"background": f"A poster ground in {color_words(palette_hexes)} with structured colour blocks.",
                                         "elements": elements},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("briefs", nargs="*", default=["robotics_day", "placement_drive", "sports_meet"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--models", nargs="*", default=["flux", "qwen"], choices=["flux", "qwen", "ideogram"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    comfy = ComfyClient()
    for name in args.briefs:
        b = load_brief(ROOT / "briefs" / name / "brief.yaml")
        brand = load_brand(b.brand)
        w, h = bg_size(CanvasSpec.from_format(b.format or "story"))
        prompt = poster_prompt(b, [brand.palette[k] for k in ("primary", "accent", "surface") if k in brand.palette])
        (OUT / f"{name}_prompt.txt").write_text(prompt, encoding="utf-8")
        tiles: list[Image.Image] = []
        for s in range(args.seeds):
            seed = 1000 + s
            for model in args.models:
                t = time.time()
                if model == "flux":
                    wf, node = flux_workflow(prompt, w, h, seed, prefix=f"ref_{name}")
                elif model == "qwen":
                    wf, node = qwen_workflow(prompt, w, h, seed, negative="", prefix=f"ref_{name}")  # no 'no text' negative: text is wanted
                else:
                    cap = ideogram_caption(b, [brand.palette[k] for k in ("primary", "accent", "surface") if k in brand.palette])
                    bad = validate_ideogram_caption(cap)
                    if bad:
                        print(f"{name} ideogram: caption off-schema, not sent: {bad}", flush=True)
                        continue
                    wf, node = ideogram_workflow(cap, w, h, seed, prefix=f"ref_{name}")
                try:
                    data = comfy.run(wf, node)
                except Exception as e:  # noqa: BLE001
                    print(f"{name} {model} seed {seed}: FAILED {str(e)[:160]}", flush=True)
                    continue
                p = OUT / f"{name}_{model}_{seed}.png"
                p.write_bytes(data)
                tiles.append(Image.open(p).convert("RGB"))
                print(f"{name} {model} seed {seed}: {time.time() - t:.0f}s -> {p.relative_to(ROOT)}", flush=True)
        if tiles:
            th = 900
            tiles = [t.resize((int(t.width * th / t.height), th)) for t in tiles]
            sheet = Image.new("RGB", (sum(t.width for t in tiles) + 12 * (len(tiles) - 1), th), "white")
            x = 0
            for t in tiles:
                sheet.paste(t, (x, 0))
                x += t.width + 12
            sheet.save(OUT / f"sheet_{name}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
