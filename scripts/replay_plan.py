"""Replay a plan through the pipeline WITHOUT an LLM planner (reproduces a run; shows the effect of code changes; works when the AI
providers are rate-limited).

Usage:
  python scripts/replay_plan.py <name> [--plan PATH] [--run-name NAME] [--art IMAGE] [--no-gen] [--critic none|auto|lfm|gemini] [--watch]

  <name>       brief folder (briefs/<name>) and, by default, the saved plan runs/<name>/plan.json
  --plan       a different plan JSON (e.g. a hand-prepared one)
  --run-name   output folder name under runs/ (default <name>_replay)
  --art        use this existing image as the generated background instead of calling ComfyUI (for when the GPU PC is off). It is run through
               a throw-away cache (data/work/stub_cache) so it can never be mistaken for real generated art.
Generated assets otherwise come from the ComfyUI cache when the requests are unchanged."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from poster_agent.assets import ComfyClient
from poster_agent.compile.vc_client import WatchConfig
from poster_agent.ingest import load_brief, scan_assets
from poster_agent.llm import ScriptedLLM, default_vlm
from poster_agent.pipeline import make_poster
from poster_agent.spec.brand import ROOT, load_brand


class ArtComfy:
    """Stands in for ComfyUI when the GPU PC is unavailable: every image request returns the supplied picture."""

    def __init__(self, art: Path):
        self.art = Path(art).read_bytes()

    def run(self, workflow, save_node):
        return self.art

    def free_vram(self):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--plan", type=Path)
    ap.add_argument("--run-name")
    ap.add_argument("--art", type=Path)
    ap.add_argument("--no-gen", action="store_true")
    ap.add_argument("--critic", choices=["none", "auto", "lfm", "gemini"], default="none")
    ap.add_argument("--watch", action="store_true", help="draw in the live VectorCraft window")
    args = ap.parse_args()
    folder = ROOT / "briefs" / args.name
    plan = json.loads((args.plan or ROOT / "runs" / args.name / "plan.json").read_text(encoding="utf-8"))
    brief = load_brief(folder / "brief.yaml")
    inv = scan_assets(folder / "assets") if (folder / "assets").exists() else {}
    run_name = args.run_name or f"{args.name}_replay"
    comfy = None if args.no_gen else (ArtComfy(args.art) if args.art else ComfyClient())
    res = make_poster(brief, load_brand(brief.brand), inv, ScriptedLLM([plan] * 6), ROOT / "runs" / run_name,
                      comfy=comfy, name=run_name, max_refine=3, cache_dir=(ROOT / "data/work/stub_cache" if args.art else None),
                      vlm=None if args.critic == "none" else default_vlm(args.critic),
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
        for i in (res.checks.issues if res.checks else []):
            print(f"     - {i.severity.upper()} {i.category.value}: {i.evidence[:200]}")


if __name__ == "__main__":
    main()
