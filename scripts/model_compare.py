"""Same layout, same measurements, three image models.

For each of 3 briefs and each model (FLUX Klein, Qwen-Image 2.1; Ideogram images are scored from files already generated), two variants:
  text   the real lines are requested at given positions            -> measured: how many requested lines were rendered correctly (OCR)
  empty  the same areas are requested as calm empty space           -> measured: lettering invented (OCR) and detail under those areas

FLUX / Qwen take prose, so each caption box is turned into a position phrase (top third, middle, bottom ...).
Usage: PYTHONPATH=. python scripts/model_compare.py [--seeds 2] [--models flux qwen]   -> eval/reference/compare.json + printed table."""
from __future__ import annotations

import argparse
import json
import sys
import time

from PIL import Image
from rapidfuzz import fuzz

from poster_agent.analysis import text_detail
from poster_agent.assets.comfy import ComfyClient
from poster_agent.assets.generate import art_problem
from poster_agent.assets.workflows import flux_workflow, qwen_workflow
from poster_agent.checks.ocr import read_lines
from poster_agent.spec import Box
from poster_agent.spec.brand import ROOT, load_brand
from poster_agent.spec.verbatim import normalize

import importlib.util
_spec = importlib.util.spec_from_file_location("ideogram_box_test", ROOT / "scripts" / "ideogram_box_test.py")
ibt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ibt)

OUT = ROOT / "eval" / "reference"


def where(bbox: list[int]) -> str:
    cy = (bbox[0] + bbox[2]) / 2
    return "across the top" if cy < 250 else "in the upper third" if cy < 400 else "in the middle" if cy < 650 else "in the lower part" if cy < 880 else "along the very bottom"


def prose(cap: dict, empty: bool) -> str:
    sd, cd = cap["style_description"], cap["compositional_deconstruction"]
    parts = [cap["high_level_description"], f"Style: {sd['aesthetics']}; lighting: {sd['lighting']}."]
    parts.append(f"Background: {cd['background']}")
    for e in cd["elements"]:
        w = where(e["bbox"])
        if e["type"] == "text" and not empty:
            parts.append(f'{w[0].upper() + w[1:]}, the text "{e["text"]}" is set as {e["desc"][0].lower() + e["desc"][1:]}.')
        elif e["type"] == "text" or e["desc"].startswith("A calm"):
            parts.append(f"{w[0].upper() + w[1:]}, a calm, smooth, softly graded open area of even tone.")
        else:
            parts.append(f"{where(e['bbox'])[0].upper() + where(e['bbox'])[1:]}, {e['desc'][0].lower() + e['desc'][1:]}.")
    return " ".join(parts)


def expected_lines(cap: dict) -> list[str]:
    return [e["text"] for e in cap["compositional_deconstruction"]["elements"] if e["type"] == "text"]


def text_score(img: Image.Image, lines: list[str]) -> tuple[int, int, str]:
    """(lines found correctly, lines requested, OCR text). A line counts when it appears in the OCR text with >= 92 % similarity."""
    ocr = normalize(" ".join(ln.text for ln in read_lines(img)))
    ok = sum(fuzz.partial_ratio(normalize(t), ocr) >= 92 for t in lines)
    return ok, len(lines), ocr


def empty_score(img: Image.Image, cap: dict) -> tuple[float, str]:
    boxes = [Box(x=e["bbox"][1], y=e["bbox"][0], w=e["bbox"][3] - e["bbox"][1], h=e["bbox"][2] - e["bbox"][0])
             for e in cap["compositional_deconstruction"]["elements"] if e["type"] == "text"]
    under = text_detail(img, boxes, 1000, 1000)
    whole = text_detail(img, [Box(x=0, y=0, w=1000, h=1000)], 1000, 1000)
    return under / max(whole, 1e-6), (art_problem(img) or "none")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--models", nargs="*", default=["flux", "qwen"])
    args = ap.parse_args()
    brand = load_brand("kct")
    pal = [brand.palette[k].upper() for k in ("primary", "accent", "surface") if k in brand.palette]
    comfy = ComfyClient()
    rows = []
    for name, (cap, w, h) in ibt.captions(pal).items():
        lines = expected_lines(cap)
        for model in args.models:
            for variant in ("text", "empty"):
                for seed in range(1, args.seeds + 1):
                    p = OUT / f"cmp_{name}_{model}_{variant}_{seed}.png"
                    if not p.exists():
                        pr = prose(cap, empty=variant == "empty")
                        wf, node = (flux_workflow(pr, w, h, seed, prefix="cmp") if model == "flux"
                                    else qwen_workflow(pr, w, h, seed, negative="", prefix="cmp"))
                        try:
                            p.write_bytes(comfy.run(wf, node))
                        except Exception as e:  # noqa: BLE001
                            print(f"{name} {model} {variant} {seed}: FAILED {str(e)[:120]}", flush=True)
                            continue
                    img = Image.open(p).convert("RGB")
                    if variant == "text":
                        ok, n, _ = text_score(img, lines)
                        rows.append(dict(brief=name, model=model, variant="text", seed=seed, ok=ok, n=n))
                        print(f"{name:16s} {model:6s} text  seed {seed}: {ok}/{n} lines correct", flush=True)
                    else:
                        ratio, lett = empty_score(img, cap)
                        rows.append(dict(brief=name, model=model, variant="empty", seed=seed, ratio=round(ratio, 2), lettering=lett))
                        print(f"{name:16s} {model:6s} empty seed {seed}: detail ratio {ratio:.2f}; lettering: {lett[:70]}", flush=True)
        # Ideogram, from the images generated earlier (same scorer)
        for seed in (1, 2):
            tp, ep = OUT / f"box_{name}_{seed}.png", OUT / f"empty_{name}_{seed}.png"
            if tp.exists():
                ok, n, _ = text_score(Image.open(tp).convert("RGB"), lines)
                rows.append(dict(brief=name, model="ideogram", variant="text", seed=seed, ok=ok, n=n))
                print(f"{name:16s} ideogram text  seed {seed}: {ok}/{n} lines correct", flush=True)
            if ep.exists():
                ratio, lett = empty_score(Image.open(ep).convert("RGB"), cap)
                rows.append(dict(brief=name, model="ideogram", variant="empty", seed=seed, ratio=round(ratio, 2), lettering=lett))
                print(f"{name:16s} ideogram empty seed {seed}: detail ratio {ratio:.2f}; lettering: {lett[:70]}", flush=True)
    (OUT / "compare.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\nSUMMARY")
    for model in ("flux", "qwen", "ideogram"):
        t = [r for r in rows if r["model"] == model and r["variant"] == "text"]
        e = [r for r in rows if r["model"] == model and r["variant"] == "empty"]
        if t:
            print(f"{model:9s} text lines correct: {sum(r['ok'] for r in t)}/{sum(r['n'] for r in t)}", end="")
        if e:
            print(f" | empty areas: lettering invented in {sum(r['lettering'] != 'none' for r in e)}/{len(e)} images, mean detail ratio "
                  f"{sum(r['ratio'] for r in e) / len(e):.2f}", end="")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
