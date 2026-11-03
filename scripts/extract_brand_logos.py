"""Interim logo extraction: render logo drawings from the brand-guide PDFs (page 6) at 600 DPI and auto-crop.

These are raster stand-ins. Replace with the institutions' original logo files (SVG/PNG) when available —
brand YAMLs reference files under brands/<school>/logos/, so swapping a file needs no code change.
Coordinates are page-6 pixel boxes measured on a 60 DPI render (1 px = 1.2 pt).
"""
from __future__ import annotations

from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
GUIDES = ROOT / "data/samples/brand_guidelines"
PX_TO_PT = 72 / 60
DPI = 600

# school -> {logo_name: (x0, y0, x1, y1) in 60-DPI page pixels}
BOXES: dict[str, dict[str, tuple[int, int, int, int]]] = {
    "KCT": {"quadrangle": (425, 235, 550, 360), "rectangle": (690, 258, 915, 338)},
    "KCLAS": {
        "primary_en": (405, 152, 685, 230),
        "primary_ta": (405, 276, 670, 382),
        "emblem": (783, 189, 886, 313),
    },
    "KCTBS": {
        "primary_en": (410, 142, 695, 255),
        "primary_ta": (410, 292, 695, 398),
        "emblem": (775, 155, 905, 372),
    },
}


def autocrop_non_white(img: Image.Image, threshold: int = 245) -> Image.Image:
    rgb = img.convert("RGB")
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, (255, 255, 255))).convert("L")
    bbox = diff.point(lambda v: 255 if v > 255 - threshold else 0).getbbox()
    return rgb.crop(bbox) if bbox else rgb


# Logos that are a solid coloured field: trim the anti-aliased edge pixels the page render leaves around them.
SOLID_FIELD = {("KCT", "quadrangle"), ("KCT", "rectangle"), ("KCLAS", "emblem")}
EDGE_TRIM_PX = 5


def main() -> None:
    for school, boxes in BOXES.items():
        doc = pymupdf.open(GUIDES / f"{school}.pdf")
        out = ROOT / "brands" / school.lower() / "logos"
        out.mkdir(parents=True, exist_ok=True)
        for name, (x0, y0, x1, y1) in boxes.items():
            clip = pymupdf.Rect(x0 * PX_TO_PT, y0 * PX_TO_PT, x1 * PX_TO_PT, y1 * PX_TO_PT)
            pix = doc[5].get_pixmap(dpi=DPI, clip=clip, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            img = autocrop_non_white(img)
            if (school, name) in SOLID_FIELD:
                img = img.crop((EDGE_TRIM_PX, EDGE_TRIM_PX, img.width - EDGE_TRIM_PX, img.height - EDGE_TRIM_PX))
            path = out / f"{name}.png"
            img.save(path)
            print(f"{school}/{name}: {img.width}x{img.height} -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
