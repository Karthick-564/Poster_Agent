"""Evaluate the agent on the evaluation briefs (eval/briefs/*.yaml): run every brief, then report what was MEASURED.

    python scripts/evaluate.py [--only name ...] [--label NAME] [--rounds 2] [--no-gen] [--no-critic] [--skip-run]

Each brief runs through run_poster.py (never published to outputs/), logs go to eval/results/<label>/logs. The result is
eval/results/<label>/results.json (per poster: produced / passed / failed checks / errors / dead space / aesthetics flags / critic note /
planner / seconds), table.md and contact.png (every poster side by side). --skip-run only re-reads existing runs and rebuilds the report.
Compare two labels with: python scripts/evaluate.py --compare labelA labelB"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BRIEFS = ROOT / "eval" / "briefs"
RESULTS = ROOT / "eval" / "results"
WORK = ROOT / "data" / "work" / "eval_briefs"


def run_one(name: str, args, logs: Path) -> float:
    folder = WORK / f"eval_{name}"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(BRIEFS / f"{name}.yaml", folder / "brief.yaml")
    shutil.rmtree(ROOT / "runs" / f"eval_{name}", ignore_errors=True)
    cmd = [sys.executable, str(ROOT / "run_poster.py"), str(folder), "--rounds", str(args.rounds), "--no-publish"]
    if args.no_gen:
        cmd.append("--no-gen")
    if args.no_critic:
        cmd.append("--no-critic")
    t = time.time()
    with open(logs / f"{name}.log", "w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    return time.time() - t


def measure(name: str, seconds: float | None) -> dict:
    rdir = ROOT / "runs" / f"eval_{name}"
    row = {"name": name, "produced": False, "passed": False, "seconds": round(seconds) if seconds else None}
    rp = rdir / "report.json"
    if not rp.exists():
        return row
    r = json.loads(rp.read_text(encoding="utf-8"))
    row["planner"] = next((x.get("llm") for x in r.get("plan_rounds", []) if x.get("stage") == "ok"), None)
    row["plan_rounds"] = len(r.get("plan_rounds", []))
    if not r.get("ok"):
        row["failure"] = "; ".join(r.get("problems", []))[:200]
        return row
    results = (r.get("checks") or {}).get("results", [])
    row.update(produced=True, passed=bool(r.get("passed")), reviewed=bool(r.get("reviewed")), kept_round=r.get("best_round"),
               failed_checks=[c["name"] for c in results if not c["passed"]],
               errors=sum(i.get("severity") == "error" for i in r.get("issues", [])),
               png=r["outputs"]["png"])
    emp = next((c for c in results if c["name"] == "emptiness"), None)
    row["dead_space"] = (emp or {}).get("detail", "")[:80]
    crit = r.get("critic") or {}
    row["critic"] = crit.get("summary") if crit.get("available") else (crit.get("note") or "n/a")
    row["layout_note"] = next((n for n in r.get("notes", []) if n.startswith("layout ")), None)
    return row


def contact_sheet(rows: list[dict], out: Path, thumb: int = 360) -> None:
    shown = [r for r in rows if r.get("png") and Path(r["png"]).exists()]
    if not shown:
        return
    cols = min(6, len(shown))
    cell_h = int(thumb * 16 / 9) + 34
    sheet = Image.new("RGB", (cols * (thumb + 10) + 10, ((len(shown) - 1) // cols + 1) * cell_h + 10), "white")
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(shown):
        im = Image.open(r["png"]).convert("RGB")
        im.thumbnail((thumb, int(thumb * 16 / 9)))
        x, y = 10 + (i % cols) * (thumb + 10), 10 + (i // cols) * cell_h
        sheet.paste(im, (x, y))
        d.text((x, y + im.height + 4), f"{r['name']}  {'PASS' if r['passed'] else 'FAIL'}", fill=(0, 0, 0))
    sheet.save(out)


def table(rows: list[dict]) -> str:
    n = len(rows)
    made, passed = sum(r["produced"] for r in rows), sum(r["passed"] for r in rows)
    lines = [f"produced {made}/{n}, passed all checks {passed}/{n}", "",
             "| brief | produced | passed | failed checks | planner | seconds |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {'yes' if r['produced'] else 'NO'} | {'yes' if r['passed'] else 'no'} | "
                     f"{', '.join(r.get('failed_checks', [])) or r.get('failure', '')} | {r.get('planner') or ''} | {r.get('seconds') or ''} |")
    return "\n".join(lines)


def compare(a: str, b: str) -> None:
    ra = {r["name"]: r for r in json.loads((RESULTS / a / "results.json").read_text(encoding="utf-8"))}
    rb = {r["name"]: r for r in json.loads((RESULTS / b / "results.json").read_text(encoding="utf-8"))}
    print(f"{'brief':22s} {a:>14s} {b:>14s}")
    for n in sorted(set(ra) | set(rb)):
        f = lambda r: "-" if r is None else ("pass" if r["passed"] else ("fail" if r["produced"] else "none"))
        print(f"{n:22s} {f(ra.get(n)):>14s} {f(rb.get(n)):>14s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--label", default=time.strftime("%Y%m%d-%H%M"))
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--no-gen", action="store_true")
    ap.add_argument("--no-critic", action="store_true")
    ap.add_argument("--skip-run", action="store_true")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    names = args.only or sorted(p.stem for p in BRIEFS.glob("*.yaml"))
    out = RESULTS / args.label
    (out / "logs").mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        secs = None if args.skip_run else run_one(name, args, out / "logs")
        rows.append(measure(name, secs))
        print(f"{name}: produced={rows[-1]['produced']} passed={rows[-1]['passed']} {rows[-1].get('failed_checks', '')}", flush=True)
    (out / "results.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "table.md").write_text(table(rows), encoding="utf-8")
    contact_sheet(rows, out / "contact.png")
    print("\n" + table(rows) + f"\n\nreport: {out}")


if __name__ == "__main__":
    main()
