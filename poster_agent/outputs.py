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


def _copy(src: Path, dst: Path, tries: int = 4) -> None:
    """shutil.copy with retries: on Windows the destination is briefly locked while an image viewer / IDE preview has it open."""
    for k in range(tries):
        try:
            shutil.copy(src, dst)
            return
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(1.0 + k)


def _free_name(outputs_dir: Path, stem: str, suffix: str = ".png") -> str:
    """First of <stem>.png, <stem>_2.png, <stem>_3.png ... that does not exist yet: earlier outputs are never overwritten."""
    n = 1
    while True:
        name = f"{stem}{suffix}" if n == 1 else f"{stem}_{n}{suffix}"
        if not (outputs_dir / name).exists():
            return name
        n += 1


def publish(run_dir: Path, outputs_dir: Path = OUTPUTS, include_failed: bool = False) -> Path | None:
    """Copy a finished run's final PNG to outputs/ under a name that is not taken yet (<name>.png, <name>_2.png, ...; existing files are
    kept as they are), plus the accepted polished version if any, and add an entry to the index."""
    run_dir = Path(run_dir)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if not report.get("ok") or (not report.get("passed") and not include_failed):
        return None  # a poster that failed its checks is not a showcase result (publish it explicitly with include_failed)
    name = run_dir.name
    outputs_dir.mkdir(parents=True, exist_ok=True)
    final = outputs_dir / _free_name(outputs_dir, name)
    _copy(run_dir / Path(report["outputs"]["png"]).name, final)
    entry = {"name": name, "image": final.name, "status": _status(report), "kept_round": report.get("best_round"),
             "published": time.strftime("%Y-%m-%d %H:%M")}
    pr = run_dir / "polish" / "polish_report.json"
    if pr.exists():
        polish = json.loads(pr.read_text(encoding="utf-8"))
        if polish.get("accepted") and polish.get("png") and Path(polish["png"]).exists():
            pol = outputs_dir / _free_name(outputs_dir, f"{final.stem}_polished")
            shutil.copy(polish["png"], pol)
            entry["polished_image"] = pol.name
    index_file = outputs_dir / "index.json"
    index = json.loads(index_file.read_text(encoding="utf-8")) if index_file.exists() else []
    index = index + [entry]
    index_file.write_text(json.dumps(sorted(index, key=lambda e: (e["name"], e["published"])), indent=1, ensure_ascii=False), encoding="utf-8")
    return final
