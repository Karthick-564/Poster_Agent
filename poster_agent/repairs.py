"""Deterministic repairs. Each records its change on the spec (`overrides`) so it survives re-layout, and returns a plain
description for the report. Nothing here asks an LLM."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from poster_agent.checks import pixels
from poster_agent.checks.runner import LARGE_TEXT_PT, LARGE_TEXT_RATIO
from poster_agent.compile.compiler import CompileResult
from poster_agent.spec import ElementOverride, Issue, PosterSpec
from poster_agent.spec.brand import BrandProfile

MIN_GAIN = 1.05  # a recolour must improve the worst-case contrast by at least 5%
MARGIN = 1.15  # a brand colour must pass with this much headroom to be preferred over plain black/white


def repair_contrast(spec: PosterSpec, brand: BrandProfile, compiled: CompileResult, issues: list[Issue]
                    ) -> tuple[list[str], list[Issue]]:
    """Recolour failing text with the brand/neutral colour that reads best over what is actually behind it.
    Returns (actions taken, issues that could not be repaired this way)."""
    img = Image.open(Path(compiled.png)).convert("RGB")
    by_id = {e.id: e for e in spec.placed}
    brand_cands = list(dict.fromkeys(brand.palette[k] for k in ("text", "primary", "surface") if k in brand.palette))
    cands = brand_cands + [c for c in ("#ffffff", "#000000") if c not in brand_cands]
    actions: list[str] = []
    unresolved: list[Issue] = []
    for issue in issues:
        fixed_any = False
        for eid in issue.element_ids:
            el = by_id.get(eid)
            if el is None or not el.color:
                continue
            box = compiled.bounds.get(eid, el.box)
            cur = pixels.hex_rgb(el.color)
            now = pixels.text_contrast(img, box, spec.canvas.export_scale, cur)
            ranked = pixels.text_color_ratios(img, box, spec.canvas.export_scale, cur, cands)
            if now["ratio"] is None or ranked is None:
                continue
            need = LARGE_TEXT_RATIO if (el.size or 0) >= LARGE_TEXT_PT else brand.min_text_contrast
            # first brand colour that passes with headroom; otherwise the best of everything (black/white last)
            color, ratio = next(((c, r) for c, r in ranked if c in brand_cands and r >= need * MARGIN), max(ranked, key=lambda t: t[1]))
            if color.lower() != el.color.lower() and ratio >= need and ratio >= now["ratio"] * MIN_GAIN:
                spec.overrides[eid] = ElementOverride(color=color)
                actions.append(f"recoloured {eid} {el.color} -> {color} (contrast {now['ratio']:.2f}:1 -> {ratio:.2f}:1)")
                fixed_any = True
        if not fixed_any:
            unresolved.append(issue)
    return actions, unresolved
