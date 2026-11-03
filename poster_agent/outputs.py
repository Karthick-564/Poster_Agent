"""The `outputs/` folder: the final poster image of every REAL agent run, and nothing else (no round images, test renders, replays or
mock-ups), so results can be shown without digging through `runs/`. Each entry is recorded in `outputs/index.json` with what was
actually measured: whether the kept poster passed the checks and whether a vision review backed it."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from poster_agent.spec.brand import ROOT

OUTPUTS = ROOT / "outputs"


def _status(report: dict) -> str:
    if not report.get("passed"):
        return "not passed (kept the best attempt; see runs/<name>/report.json)"
    return "passed checks and vision review" if report.get("reviewed") else "passed checks only (no vision review)"


def publish(run_dir: Path, outputs_dir: Path = OUTPUTS) -> Path | None:
    """Copy a finished run's final PNG to outputs/<name>.png (plus the accepted polished version, if any) and update the index."""
    run_dir = Path(run_dir)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if not report.get("ok"):
        return None
    name = run_dir.name
    outputs_dir.mkdir(parents=True, exist_ok=True)
    final = outputs_dir / f"{name}.png"
    shutil.copy(run_dir / Path(report["outputs"]["png"]).name, final)
    entry = {"name": name, "image": final.name, "status": _status(report), "kept_round": report.get("best_round"),
             "published": time.strftime("%Y-%m-%d %H:%M")}
    pr = run_dir / "polish" / "polish_report.json"
    if pr.exists():
        polish = json.loads(pr.read_text(encoding="utf-8"))
        if polish.get("accepted") and polish.get("png") and Path(polish["png"]).exists():
            shutil.copy(polish["png"], outputs_dir / f"{name}_polished.png")
            entry["polished_image"] = f"{name}_polished.png"
    index_file = outputs_dir / "index.json"
    index = json.loads(index_file.read_text(encoding="utf-8")) if index_file.exists() else []
    index = [e for e in index if e["name"] != name] + [entry]
    index_file.write_text(json.dumps(sorted(index, key=lambda e: e["name"]), indent=1, ensure_ascii=False), encoding="utf-8")
    return final
