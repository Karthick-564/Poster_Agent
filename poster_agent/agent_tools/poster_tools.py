"""Poster helper tools for a model that designs directly in VectorCraft (stdio MCP server, run as `python -m poster_agent.agent_tools.poster_tools`).

This server never draws: VectorCraft's own native MCP (`vectorcraft-cli mcp`) does all the drawing. These tools only MEASURE and CHECK, so
the model does not have to guess pixels: font metrics and text wrapping, where the art is calm, contrast over the art, brand facts, and
the same kind of verbatim / legibility checks the pipeline uses. Every number returned is a measurement of real input."""
from __future__ import annotations

import json
import re
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from PIL import Image

from poster_agent.analysis import analyse_image
from poster_agent.checks import ocr, pixels
from poster_agent.checks.emptiness import measure_emptiness
from poster_agent.ingest import load_brief
from poster_agent.layout.fonts import TextMeasurer
from poster_agent.spec import Box, CanvasSpec
from poster_agent.spec.brand import ROOT, load_brand
from poster_agent.spec.formats import FORMATS, MM_TO_PT

mcp = MCPServer("poster-tools")
_measurer = TextMeasurer()


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)


@mcp.tool()
def canvas_info(format: str, brand: str = "kct") -> str:
    """Canvas size in points for a format (story | fourfive | square | a3 | a4), the brand's safe box and the export pixel size."""
    b = load_brand(brand)
    c = CanvasSpec.from_format(format, b.safe_margin_pct)
    s = c.safe_box
    return _j({"format": c.format, "width_pt": c.width, "height_pt": c.height, "export_scale": c.export_scale,
               "safe_box": {"x": s.x, "y": s.y, "w": s.w, "h": s.h}, "formats": sorted(FORMATS)})


@mcp.tool()
def brand_info(brand: str = "kct") -> str:
    """Brand facts: palette, font roles (family + style), logo files (absolute path, kind, aspect, minimum width, whether the artwork carries
    its own solid field), logo rules in points, forbidden grounds and the default footer lines. Follow these exactly."""
    b = load_brand(brand)
    rules = b.logo_rules
    return _j({
        "name": b.name, "palette": b.palette, "fonts": {k: {"family": v.family, "style": v.style} for k, v in b.fonts.items()},
        "logos": {k: {"path": str(ROOT / v.path), "kind": v.kind, "language": v.language, "field": v.field, "aspect_w_over_h": round(v.aspect, 3),
                      "min_width_pct_of_canvas": v.min_width_pct, "solid_field": v.solid_field} for k, v in b.logos.items()},
        "logo_rules": {"clear_space_pt": round(rules.clear_space_mm * MM_TO_PT, 1), "max_width_pct": rules.max_width_pct,
                       "max_height_pct": rules.max_height_pct, "forbidden": rules.forbidden},
        "forbidden_grounds": b.forbidden_grounds, "safe_margin_pct": b.safe_margin_pct,
    })


@mcp.tool()
def measure_text(text: str, font_family: str, font_style: str, size_pt: float, width_pt: float, tracking: float = 0.0,
                 leading: float = 1.2) -> str:
    """How `text` wraps in a box `width_pt` wide at `size_pt` (the same font metrics the layout engine uses, within about 1.5% of
    VectorCraft): the wrapped lines, the widest line, the block height (lines * size * leading) and whether any single word is too wide.
    tracking is in 1/1000 em. Use it before placing text so boxes are sized to the text, not guessed."""
    lines = _measurer.wrap(text, font_family, font_style, size_pt, width_pt, tracking)
    widest = _measurer.widest(lines, font_family, font_style, size_pt, tracking)
    return _j({"lines": lines, "line_count": len(lines), "widest_pt": round(widest, 1), "block_height_pt": round(len(lines) * size_pt * leading, 1),
               "fits_width": widest <= width_pt})


@mcp.tool()
def fit_text(text: str, font_family: str, font_style: str, width_pt: float, max_lines: int = 3, max_size_pt: float = 200.0,
             min_size_pt: float = 10.0, tracking: float = 0.0, leading: float = 1.2) -> str:
    """The LARGEST font size (pt) at which `text` fits `width_pt` in at most `max_lines` lines, with the wrapped lines and block height.
    Use it to make a headline as big as the space allows."""
    lo, hi, best = min_size_pt, max_size_pt, None
    for _ in range(24):
        mid = (lo + hi) / 2
        lines = _measurer.wrap(text, font_family, font_style, mid, width_pt, tracking)
        ok = len(lines) <= max_lines and _measurer.widest(lines, font_family, font_style, mid, tracking) <= width_pt
        if ok:
            best, lo = (mid, lines), mid
        else:
            hi = mid
    if best is None:
        return _j({"fits": False, "reason": f"does not fit {width_pt}pt in {max_lines} lines even at {min_size_pt}pt"})
    size, lines = best
    return _j({"fits": True, "size_pt": round(size - 0.25, 1), "lines": lines, "block_height_pt": round(len(lines) * size * leading, 1)})


@mcp.tool()
def art_map(image_path: str, canvas_width_pt: float, canvas_height_pt: float) -> str:
    """Where a background picture is calm and where it is busy or has its subject, in canvas points (the picture is cover-fitted to the
    canvas as VectorCraft will draw it). Returns a coarse text map (rows top to bottom; '.' calm, 'o' some detail, '#' busy,
    'S' subject), the subject centre, and the largest calm rectangles where text can sit."""
    with Image.open(_abs(image_path)) as im:
        im = im.convert("RGB")
        k = max(canvas_width_pt / im.width, canvas_height_pt / im.height)
        im = im.resize((max(int(im.width * k), 1), max(int(im.height * k), 1)))
        x0, y0 = (im.width - int(canvas_width_pt)) // 2, (im.height - int(canvas_height_pt)) // 2
        im = im.crop((x0, y0, x0 + int(canvas_width_pt), y0 + int(canvas_height_pt)))
    art = analyse_image(im, canvas_width_pt, canvas_height_pt)
    rows, cols = 16, 10
    lines = []
    for r in range(rows):
        row = ""
        for c in range(cols):
            box = Box(x=c * canvas_width_pt / cols, y=r * canvas_height_pt / rows, w=canvas_width_pt / cols, h=canvas_height_pt / rows)
            b, s = art.busy_in(box), art.subject_in(box)
            row += "S" if s > 0.6 else ("#" if b > 0.55 else ("o" if b > 0.25 else "."))
        lines.append(row)
    calm = []
    for w_frac in (0.9, 0.6):
        for h_frac in (0.35, 0.22, 0.14):
            best = None
            bw, bh = canvas_width_pt * w_frac, canvas_height_pt * h_frac
            for yy in range(0, int(canvas_height_pt - bh) + 1, int(canvas_height_pt / 40) or 1):
                for xx in range(0, int(canvas_width_pt - bw) + 1, int(canvas_width_pt / 20) or 1):
                    bx = Box(x=xx, y=yy, w=bw, h=bh)
                    cost = art.busy_in(bx) + 2 * art.subject_in(bx)
                    if best is None or cost < best[0]:
                        best = (cost, bx)
            calm.append({"x": round(best[1].x), "y": round(best[1].y), "w": round(bw), "h": round(bh), "cost": round(best[0], 2)})
    cen = art.subject_centroid()
    return _j({"grid": "\n".join(lines), "grid_note": f"{rows} rows x {cols} cols, each cell {canvas_width_pt / cols:.0f} x {canvas_height_pt / rows:.0f} pt",
               "subject_center_pt": [round(cen[0]), round(cen[1])] if cen else None, "calm_rectangles_lowest_cost_for_each_size": calm})


@mcp.tool()
def contrast_on_art(image_path: str, canvas_width_pt: float, canvas_height_pt: float, x: float, y: float, w: float, h: float,
                    text_color_hex: str) -> str:
    """WCAG contrast ratio of a text colour against the picture pixels in the box (x, y, w, h in canvas points), at the worst 10th/90th
    percentile of the pixels behind the text. Needs 3.0 for large text and 4.5 for body text. image_path can be the background art
    (cover-fitted) or an exported render of the poster so far."""
    with Image.open(_abs(image_path)) as im:
        im = im.convert("RGB")
        k = max(canvas_width_pt / im.width, canvas_height_pt / im.height)
        im = im.resize((max(int(im.width * k), 1), max(int(im.height * k), 1)))
        x0, y0 = (im.width - int(canvas_width_pt)) // 2, (im.height - int(canvas_height_pt)) // 2
        im = im.crop((x0, y0, x0 + int(canvas_width_pt), y0 + int(canvas_height_pt)))
    res = pixels.text_contrast(im, Box(x=x, y=y, w=w, h=h), 1.0, pixels.hex_rgb(text_color_hex))
    ratio = res["ratio"]
    return _j({"ratio": None if ratio is None else round(ratio, 2), "ok_for_large_text": bool(ratio and ratio >= 3.0),
               "ok_for_body_text": bool(ratio and ratio >= 4.5), "background_median_luminance": res.get("bg_median_lum")})


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


@mcp.tool()
def audit_text(svg_path: str, brief_path: str) -> str:
    """Verbatim audit of an exported SVG: every text string in the poster must come from the brief (copy exact words). Returns the strings
    found, any string whose words are NOT in the brief (invented or altered copy), and which brief fields are missing from the poster.
    Export the poster as SVG first (VectorCraft export, format svg)."""
    svg = _abs(svg_path).read_text(encoding="utf-8", errors="ignore")
    found = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", m)).strip() for m in re.findall(r"<text\b[^>]*>(.*?)</text>", svg, flags=re.S)]
    found = [t for t in found if t]
    brief = load_brief(_abs(brief_path))
    corpus = _norm(" ".join(brief.corpus()))
    cwords = set(corpus.split())
    foreign = [t for t in found if any(wd not in cwords for wd in _norm(t).split())]
    joined = _norm(" ".join(found))
    missing = [t for _, t in brief.required_fields() if _norm(t) and _norm(t) not in joined]
    return _j({"text_strings": found, "not_from_brief": foreign, "brief_fields_missing_from_poster": missing,
               "verbatim_ok": not foreign and not missing})


@mcp.tool()
def check_render(png_path: str, brief_path: str, format: str, brand: str = "kct") -> str:
    """Measure an exported PNG of the poster: size matches the format, every brief text field can be READ by OCR, the empty space
    (largest empty region, tallest empty band), and whether the institution logo's artwork appears. Use it after each export; fix what it
    reports before finishing. Only reports what was actually measured."""
    img = Image.open(_abs(png_path)).convert("RGB")
    b = load_brand(brand)
    canvas = CanvasSpec.from_format(format, b.safe_margin_pct)
    scale = img.width / canvas.width
    out: dict = {"png_size": [img.width, img.height],
                 "size_matches_format": abs(img.width / img.height - canvas.width / canvas.height) < 0.01}
    brief = load_brief(_abs(brief_path))
    lines = ocr.read_lines(img)
    corpus = " ".join(ln.text for ln in lines)
    miss = []
    for path, text in brief.required_fields():
        if not ocr.verifiable(text):
            continue
        if ocr.match_score(text, corpus) < 80:
            miss.append({"field": path, "text": text})
    out["ocr_missing_brief_text"] = miss
    emp = measure_emptiness(img, scale)
    out["empty_space"] = {"largest_empty_region_pct": round(100 * emp.largest_fraction, 1), "tallest_empty_band_pct_of_height": round(100 * emp.longest_band_fraction, 1),
                          "total_empty_pct": round(100 * emp.empty_fraction, 1), "limits": "region <= 14, band <= 3.5, total <= 24"}
    out["measured"] = "OCR over the whole image, pixel emptiness, size. Contrast and logo rules are not measured here (use contrast_on_art and brand_info)."
    return _j(out)


if __name__ == "__main__":
    mcp.run()
