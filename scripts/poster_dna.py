"""Measure HOW posters are built: text sizes, hierarchy ratio, margins, vertical rhythm, fill. Runs OCR over a folder of posters and writes
a table, so the layout rules come from measurements of the designers' own work instead of guesses.

    python scripts/poster_dna.py designers   -> data/work/dna_designers.json
    python scripts/poster_dna.py ours        -> data/work/dna_ours.json   (recent renders listed below)
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poster_agent.checks import ocr  # noqa: E402

OURS = ["outputs/koncept_symposium_2.png", "runs/koncept_symposium/koncept_symposium.png", "runs/sports_meet/sports_meet.png",
        "runs/green_campus_talk/green_campus_talk.png", "runs/eval_alumni_meet/eval_alumni_meet.png", "runs/eval_seminar_energy/eval_seminar_energy.png",
        "runs/eval_workshop_data/eval_workshop_data.png", "runs/eval_cultural_fest/eval_cultural_fest.png"]


def measure(path: Path) -> dict | None:
    im = Image.open(path).convert("RGB")
    W0, H0 = im.size
    k = 1100 / max(W0, H0)
    im = im.resize((int(W0 * k), int(H0 * k)))
    W, H = im.size
    lines = [ln for ln in ocr.read_lines(im) if ln.conf >= 0.5 and len(ocr.alnum(ln.text)) >= 2]
    if len(lines) < 3:
        return None
    hs = np.array([(ln.y1 - ln.y0) / H for ln in lines])  # line height as a share of the poster height
    order = np.argsort(-hs)
    head = hs[order[: max(1, len(hs) // 8)]]
    body = np.median(hs)
    x0s = np.array([ln.x0 / W for ln in lines]); x1s = np.array([ln.x1 / W for ln in lines])
    ys = sorted((ln.y0 / H, ln.y1 / H) for ln in lines)
    cover = np.zeros(200, bool)
    for a, b in ys:
        cover[int(a * 200): int(np.ceil(b * 200))] = True
    gaps = [(ys[i + 1][0] - ys[i][1]) for i in range(len(ys) - 1) if ys[i + 1][0] - ys[i][1] > 0.004]
    sizes = sorted({round(float(h) * 200) / 200 for h in hs}, reverse=True)  # distinct size classes at 0.5% resolution
    return {
        "file": path.name[:44], "lines": len(lines), "headline_h_pct": round(100 * float(head.max()), 1), "median_h_pct": round(100 * float(body), 2),
        "hierarchy_ratio": round(float(head.max() / body), 1), "size_classes": len(sizes),
        "left_margin_pct": round(100 * float(np.percentile(x0s, 10)), 1), "right_margin_pct": round(100 * float(1 - np.percentile(x1s, 90)), 1),
        "rows_with_text_pct": round(100 * float(cover.mean())), "median_gap_pct": round(100 * float(st.median(gaps)), 2) if gaps else 0,
        "text_top_pct": round(100 * ys[0][0]), "text_bottom_pct": round(100 * ys[-1][1]),
        "aspect": round(W0 / H0, 2),
    }


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "designers"
    if which == "designers":
        files = sorted(p for p in (ROOT / "data/samples/designer_posters").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    else:
        files = [ROOT / p for p in OURS if (ROOT / p).exists()]
    rows = []
    for f in files:
        try:
            r = measure(f)
        except Exception as e:  # one unreadable file must not stop the survey
            print("skip", f.name, str(e)[:60]); continue
        if r:
            rows.append(r); print(r["file"], r["headline_h_pct"], r["median_h_pct"], r["hierarchy_ratio"], flush=True)
    (ROOT / "data/work" / f"dna_{which}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    keys = ["headline_h_pct", "median_h_pct", "hierarchy_ratio", "size_classes", "left_margin_pct", "right_margin_pct", "rows_with_text_pct", "median_gap_pct"]
    print("\n== median (and 25-75 percentile) over", len(rows), which)
    for k in keys:
        v = [r[k] for r in rows]
        print(f"{k:20s} {st.median(v):6.1f}   {np.percentile(v, 25):6.1f} - {np.percentile(v, 75):6.1f}")


if __name__ == "__main__":
    main()
