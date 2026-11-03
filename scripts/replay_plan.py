"""Replay a saved plan through the pipeline without calling an LLM planner (reproduces a run; shows the effect of code changes).
Usage: python scripts/replay_plan.py <run name> [--no-gen]      e.g. robotics_day  (uses runs/<name>/plan.json + briefs/<name>/)
Output goes to runs/<name>_replay/. Generated assets come from the cache when the requests are unchanged."""
from __future__ import annotations

import argparse
import json

from poster_agent.assets import ComfyClient
from poster_agent.compile.vc_client import WatchConfig
from poster_agent.ingest import load_brief, scan_assets
from poster_agent.llm import ScriptedLLM
from poster_agent.pipeline import make_poster
from poster_agent.spec.brand import ROOT, load_brand


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--no-gen", action="store_true")
    ap.add_argument("--watch", action="store_true", help="draw in the live VectorCraft window")
    args = ap.parse_args()
    folder = ROOT / "briefs" / args.name
    plan = json.loads((ROOT / "runs" / args.name / "plan.json").read_text(encoding="utf-8"))
    brief = load_brief(folder / "brief.yaml")
    inv = scan_assets(folder / "assets") if (folder / "assets").exists() else {}
    res = make_poster(brief, load_brand(brief.brand), inv, ScriptedLLM([plan] * 6), ROOT / "runs" / f"{args.name}_replay",
                      comfy=None if args.no_gen else ComfyClient(), name=args.name, max_refine=3,
                      watch=WatchConfig(pace=0.15, launch=False) if args.watch else None)
    for r in res.plan.rounds:
        print(f"plan round {r.round}: {r.stage}", *[f"\n   - {p}" for p in r.problems])
    for g in res.generated:
        print(f"generated {g.kind}: {g.model} seed={g.seed} cached={g.cached} rejected={len(g.rejected)}")
    for n in res.notes:
        print("NOTE:", n)
    for rd in res.rounds:
        print(f"round {rd.n}: {rd.errors} error(s), dead space {rd.empty_fraction:.0%}, passed={rd.passed}", *[f"\n     -> {a}" for a in rd.actions])
    if res.ok:
        print(f"{'PASSED' if res.passed else 'NOT PASSED'} (kept round {res.best_round}) ->", res.compiled.png)
        for c in (res.checks.results if res.checks else []):
            print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")


if __name__ == "__main__":
    main()
