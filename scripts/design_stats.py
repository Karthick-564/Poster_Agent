"""How a poster is DESIGNED, in numbers: headline size, text column, tiers of type. Run on model designs and on real hand-made posters.
Measures from OCR boxes only (garbled words still have boxes), so it reads layout, not spelling.
Usage: PYTHONPATH=. python scripts/design_stats.py <image files or folders...>"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from poster_agent.checks.ocr import read_lines


def stats(path: Path) -> dict | None:
    img = Image.open(path).convert("RGB")
    img.thumbnail((1100, 1500))
    W, H = img.size
    ls = [l for l in read_lines(img) if (l.y1 - l.y0) > 0.006 * H and l.conf >= 0.5]
    if len(ls) < 3:
        return None
    hs = np.array([l.y1 - l.y0 for l in ls]) / H
    top = ls[int(np.argmax(hs))]
    big = [l for l, h in zip(ls, hs) if h >= 0.6 * hs.max()]  # the headline lines
    block = (max(l.y1 for l in big) - min(l.y0 for l in big)) / H
    tiers = len({round(float(h) / 0.01) for h in hs}) if len(hs) else 0
    sizes = sorted(hs, reverse=True)
    second = next((s for s in sizes if s < 0.6 * hs.max()), sizes[-1])
    xs0, xs1 = np.array([l.x0 for l in ls]) / W, np.array([l.x1 for l in ls]) / W
    return {"name": path.name[:34], "headline_line_h": round(float(hs.max()), 3), "headline_block_h": round(block, 3),
            "headline_to_next": round(float(hs.max() / max(second, 1e-6)), 1), "left_margin": round(float(np.percentile(xs0, 10)), 2),
            "text_right_edge": round(float(np.percentile(xs1, 90)), 2), "text_top": round(float(min(l.y0 for l in ls) / H), 2),
            "text_bottom": round(float(max(l.y1 for l in ls) / H), 2), "lines": len(ls), "tiers": tiers}


def main() -> int:
    files: list[Path] = []
    for a in sys.argv[1:]:
        p = Path(a)
        files += sorted(p.rglob("*.jpg")) + sorted(p.rglob("*.png")) if p.is_dir() else [p]
    rows = [r for r in (stats(f) for f in files) if r]
    keys = ["headline_line_h", "headline_block_h", "headline_to_next", "left_margin", "text_right_edge", "text_top", "text_bottom", "lines", "tiers"]
    print(f"{'poster':36s}" + "".join(f"{k[:11]:>12s}" for k in keys))
    for r in rows:
        print(f"{r['name']:36s}" + "".join(f"{r[k]:>12}" for k in keys))
    if rows:
        print(f"{'MEDIAN':36s}" + "".join(f"{float(np.median([r[k] for r in rows])):>12.2f}" for k in keys))
    return 0


if __name__ == "__main__":
    sys.exit(main())
