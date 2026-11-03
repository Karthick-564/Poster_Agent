"""Make a poster from a brief folder (brief.yaml + assets/). Usage:
    python run_poster.py briefs/kciiri_annual_day [--no-gen] [--no-critic] [--rounds 3] [--examples engineers_day industry_conclave]

Rare, opt-in modes for hands-on work in the VectorCraft desktop app:
    --watch [--pace 0.35] [--port 7979]   draw in the live app window so you can watch; the finished poster is left open for editing
    --finalize runs/<name> [--no-ocr]      after you edit and save it, re-export and re-run the checks on what you ended up with
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from poster_agent.assets import ComfyClient
from poster_agent.compile.vc_client import WatchConfig
from poster_agent.finalize import finalize
from poster_agent.ingest import load_brief, scan_assets
from poster_agent.llm import default_llm, default_vlm
from poster_agent.pipeline import make_poster
from poster_agent.plan.examples import example_plans
from poster_agent.spec.brand import ROOT, load_brand


def run_finalize(target: Path, use_ocr: bool) -> None:
    res = asyncio.run(finalize(target, use_ocr=use_ocr))
    print(f"re-exported -> {res.png}")
    if res.changes:
        print("what you changed:")
        for c in res.changes:
            print(f"   {c.kind:12s} {c.id}: {c.detail}")
    else:
        print("no edits detected (the document matches the generated poster)")
    for c in res.report.results:
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
    for i in res.report.issues + res.extra_issues:
        print(f"     - {i.severity.upper()} {i.category.value}: {i.evidence[:220]}")
    print("RESULT:", "your version passes the brand checks" if res.ok else "your version breaks a check (see the errors above)")


def run_polish(target: Path, use_ocr: bool) -> None:
    from poster_agent.llm import default_vlm
    from poster_agent.polish import polish
    res = asyncio.run(polish(target, default_vlm(), use_ocr=use_ocr))
    print("POLISH:", res.reason)
    if res.rationale:
        print("  rationale:", res.rationale)
    for op in res.ops_applied:
        print("  applied:", op)
    for r in res.rejected:
        print("  rejected:", r)
    if res.accepted:
        print("  polished poster ->", res.png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path, help="brief folder; with --finalize: the run folder (runs/<name>) or its .vectorcraft file")
    ap.add_argument("--no-gen", action="store_true", help="do not use ComfyUI (no generated art)")
    ap.add_argument("--no-critic", action="store_true", help="skip the vision critic")
    ap.add_argument("--rounds", type=int, default=3, help="max refine rounds after the first render")
    ap.add_argument("--examples", nargs="*", default=["engineers_day"])
    ap.add_argument("--watch", action="store_true", help="draw in the live VectorCraft app window and leave the result open for editing")
    ap.add_argument("--pace", type=float, default=0.35, help="seconds between visible steps in --watch mode")
    ap.add_argument("--port", type=int, default=7979, help="VectorCraft control port for --watch")
    ap.add_argument("--finalize", action="store_true", help="re-export and re-check a poster you edited by hand")
    ap.add_argument("--no-ocr", action="store_true", help="with --finalize: skip the (slower) OCR checks")
    ap.add_argument("--polish", action="store_true", help="after the run, let the vision LLM propose a few allow-listed fine touches (kept only if checks do not worsen)")
    ap.add_argument("--publish", action="store_true", help="copy the final image of an existing run folder (runs/<name>) to outputs/")
    ap.add_argument("--polish-only", action="store_true", help="polish an existing run folder (runs/<name>) without regenerating")
    args = ap.parse_args()

    if args.finalize:
        run_finalize(args.folder, use_ocr=not args.no_ocr)
        return
    if args.publish:
        from poster_agent.outputs import publish
        print("final image ->", publish(args.folder))
        return
    if args.polish_only:
        run_polish(args.folder, use_ocr=not args.no_ocr)
        return

    folder = args.folder.resolve()
    brief = load_brief(folder / "brief.yaml")
    brand = load_brand(brief.brand)
    inventory = scan_assets(folder / "assets") if (folder / "assets").exists() else {}
    comfy = None if args.no_gen else ComfyClient()
    out = ROOT / "runs" / folder.name
    watch = WatchConfig(port=args.port, pace=args.pace) if args.watch else None
    res = make_poster(brief, brand, inventory, default_llm(), out, comfy=comfy, examples=example_plans(args.examples),
                      name=folder.name, vlm=None if args.no_critic else default_vlm(), max_refine=args.rounds, watch=watch)

    for r in res.plan.rounds:
        print(f"plan round {r.round}: {r.stage} ({r.llm})", *[f"\n   - {p}" for p in r.problems])
    for g in res.generated:
        print(f"generated {g.kind}: {g.model} seed={g.seed} {g.seconds:.0f}s cached={g.cached} -> {g.path}")
    for n in res.notes:
        print("NOTE:", n)
    if not res.ok:
        print("FAILED:", *[f"\n   - {p}" for p in res.plan.problems])
        return
    for rd in res.rounds:
        crit = rd.critic
        cs = "n/a" if not crit else ("unavailable" if not crit.available else f"{crit.score}/10")
        print(f"round {rd.n}: {rd.errors} error(s), critic {cs}, passed={rd.passed}")
        for a in rd.actions:
            print("     ->", a)
    verdict = "NOT PASSED" if not res.passed else ("PASSED" if res.reviewed else "PASSED CHECKS ONLY (no vision review)")
    print(f"{verdict} (kept round {res.best_round}) ->", res.compiled.png)
    if res.checks:
        for c in res.checks.results:
            print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
    if res.critic and res.critic.available:
        print("critic:", res.critic.summary)
    elif res.critic:
        print("critic:", res.critic.note)
    for i in (res.checks.issues if res.checks else []) + (res.critic.issues if res.critic else []):
        print(f"     - {i.severity.upper()} {i.source}/{i.category.value}: {i.evidence[:200]}")
    if args.polish and res.ok:
        run_polish(out, use_ocr=True)
    if res.ok:
        from poster_agent.outputs import publish
        print("final image ->", publish(out))
    if watch:
        print(f"\nThe poster is open in VectorCraft ({res.compiled.vectorcraft.name}). Edit it, save with Ctrl+S, then run:\n"
              f"    python run_poster.py --finalize runs/{folder.name}")


if __name__ == "__main__":
    main()
