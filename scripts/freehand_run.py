"""Experiment: a fresh agent session designs the poster DIRECTLY in VectorCraft through the native MCP, with no layout engine.

    python scripts/freehand_run.py <brief.yaml> [--art IMAGE] [--name NAME] [--model sonnet] [--max-turns 70]

The session gets: the brief, the background art (if any), the brand through our helper tools (poster_agent/agent_tools: measure text, read the
art map, contrast, verbatim audit, render check) and VectorCraft's native MCP for all drawing. Output: runs/freehand_<name>/ with poster.png /
poster.pdf / poster.vectorcraft / poster.svg and agent.log (the session's tool use). Nothing is published to outputs/. Afterwards the SAME
measurements are applied to it as to the pipeline's posters (`measure()` below) so the two are comparable."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poster_agent.compile.vc_client import cli_path, session_args  # noqa: E402
from poster_agent.llm.cli_llm import NEUTRAL_CWD, cli_path as planner_cli_path  # noqa: E402

SYSTEM = """You are a senior graphic designer making one event poster for an educational institution, directly in the VectorCraft vector
design application through its tools. There is no layout engine: YOU decide where everything goes, and you check your own work.

HARD RULES (a poster that breaks one is rejected)
1. Every word on the poster comes from the brief file, copied exactly (you may change letter case). Add no words: no invented taglines, no
   placeholder text, no labels. Include every field of the brief (title, dates, times, venue, lines, call to action, contacts).
2. The institution logo is a supplied file (brand_info gives its path): place it unaltered, proportional, never recoloured / cropped /
   effected, no larger than the brand's maximum, with the brand's clear space around it, on a flat field (or use a logo whose artwork carries
   its own solid field). Do not draw or redraw a logo.
3. Use only the brand's palette colours and font families (brand_info). Keep everything inside the safe box except full-bleed art or bands.
4. The background picture (if given) is placed as it is, cover-fitted to the canvas; never altered by you beyond placement and opacity/overlays.

DESIGN BRIEF: clean, professional, institutional. Not ornamental: no dots, glows, rings or random shapes. Fill space with SCALE and
STRUCTURE: a dominant title (clearly the largest thing, at least 1.5x the next element), a clear hierarchy, one strong alignment edge, generous
consistent spacing, and details grouped by meaning (date/time/venue together). Put text where the picture is calm (art_map) and away from its
subject; make sure text stays readable on the picture (contrast_on_art: 3.0 for large text, 4.5 for body), using a plain colour band or a
soft gradient only if you must. A footer band for contact / call-to-action lines is welcome. Vary the composition to suit THIS brief and
THIS picture; do not default to logo-centre-title-centre-details.

WORKFLOW
1. Call brand_info, canvas_info (format from the brief) and, if there is art, art_map. Think about the composition (do not write it out).
2. Create the document in VectorCraft at the format's size in points, place the art, then the text and logo. Size every text box from
   measure_text / fit_text (they use the real fonts), never by guessing. Give objects meaningful names (title, date, logo, ...).
3. Export a PNG to poster.png in the working folder and LOOK at it with the Read tool. Run check_render. Fix what you see and what it reports
   (overflow, collisions, weak hierarchy, unreadable text, empty regions). You have at most 3 revision rounds after the first export.
4. Export poster.svg and run audit_text on it: every string must be from the brief. Fix any that is not.
5. Final exports in the working folder: poster.png, poster.pdf, poster.vectorcraft (and poster.svg). Then reply with 3 short lines: what the
   composition is, what you fixed, and anything you could not satisfy. Do not claim a check passed unless you ran it and saw it pass.

VECTORCRAFT TIPS (learned the hard way): the document is in points; area text is a frame (give width and height, check overflow);
add_text has no font weight parameter, so use run_command text.create with a style when you need a specific weight/style; command_batch can
chain steps with $N.id references; to mask an image the clipping path must be topmost; stroke.set uses `weight`; export takes the scale at the
top level. Paths must be inside the working folder."""


def prepare(brief: Path, art: Path | None, name: str) -> Path:
    work = ROOT / "runs" / f"freehand_{name}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    shutil.copy(brief, work / "brief.yaml")
    if art:
        shutil.copy(art, work / f"background{art.suffix}")
    mcp = {"mcpServers": {
        "vectorcraft": {"command": str(cli_path()), "args": session_args([ROOT], None)},
        "postertools": {"command": sys.executable, "args": ["-m", "poster_agent.agent_tools.poster_tools"], "env": {"PYTHONPATH": str(ROOT)},
                        "cwd": str(ROOT)}}}
    (work / "mcp.json").write_text(json.dumps(mcp, indent=1), encoding="utf-8")
    (work / "system.txt").write_text(SYSTEM, encoding="utf-8")
    return work


def run(work: Path, model: str, max_turns: int, timeout: float) -> float:
    rel = work.relative_to(ROOT).as_posix()
    art = next(iter(work.glob("background.*")), None)
    user = (f"Working folder (inside the project, VectorCraft may write here): {rel}\nBrief file: {rel}/brief.yaml\n"
            + (f"Background art file: {rel}/{art.name} (place it full-bleed, cover-fitted)\n" if art else "No background art: use a plain ground.\n")
            + f"Brand: kct. Write all exports to {rel}/poster.* (png, pdf, vectorcraft, svg).")
    cmd = [planner_cli_path(), "-p", "--system-prompt-file", str(work / "system.txt"), "--mcp-config", str(work / "mcp.json"), "--strict-mcp-config",
           "--allowedTools", "mcp__vectorcraft", "mcp__postertools", "Read", "--no-session-persistence", "--setting-sources", "",
           "--model", model, "--max-turns", str(max_turns), "--output-format", "stream-json", "--verbose"]
    t = time.time()
    with open(work / "agent.log", "w", encoding="utf-8") as f:
        try:
            subprocess.run(cmd, input=user, text=True, encoding="utf-8", stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, timeout=timeout)
        except subprocess.TimeoutExpired:
            f.write("\n[timeout]\n")
    return time.time() - t


def summarize(work: Path) -> dict:
    """What the session did, from its log: tool calls by name, turns, final text, cost if reported."""
    calls, final, cost = {}, "", None
    for ln in (work / "agent.log").read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            ev = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            for c in ev.get("message", {}).get("content", []):
                if c.get("type") == "tool_use":
                    calls[c["name"]] = calls.get(c["name"], 0) + 1
        if ev.get("type") == "result":
            final, cost = str(ev.get("result", ""))[:600], ev.get("total_cost_usd")
    return {"tool_calls": calls, "total_calls": sum(calls.values()), "final": final, "cost_usd": cost}


def measure(work: Path) -> dict:
    """The same kind of measurements as the pipeline's checks, on the files the session produced."""
    from poster_agent.agent_tools import poster_tools as pt
    out = {"files": {k: (work / f"poster.{k}").exists() for k in ("png", "pdf", "vectorcraft", "svg")}}
    brief = work / "brief.yaml"
    import yaml
    fmt = yaml.safe_load(brief.read_text(encoding="utf-8")).get("format", "story")
    if out["files"]["png"]:
        out["render"] = json.loads(pt.check_render(str(work / "poster.png"), str(brief), fmt))
    if out["files"]["svg"]:
        out["text_audit"] = json.loads(pt.audit_text(str(work / "poster.svg"), str(brief)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("brief", type=Path)
    ap.add_argument("--art", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--max-turns", type=int, default=70)
    ap.add_argument("--timeout", type=float, default=1500)
    ap.add_argument("--measure-only", action="store_true")
    args = ap.parse_args()
    name = args.name or args.brief.stem
    work = ROOT / "runs" / f"freehand_{name}"
    secs = None
    if not args.measure_only:
        work = prepare(args.brief.resolve(), args.art.resolve() if args.art else None, name)
        secs = run(work, args.model, args.max_turns, args.timeout)
    res = {"name": name, "seconds": round(secs) if secs else None, **summarize(work), **measure(work)}
    (work / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
