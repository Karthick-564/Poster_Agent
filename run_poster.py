"""Make a poster from a brief folder (brief.yaml + assets/). Usage:
    python run_poster.py --prompt "Generative AI workshop, 12 Dec 2026, Seminar Hall 2 ..." [--brand kct] [--format story] [--name my_event]
    python run_poster.py briefs/kciiri_annual_day [--no-gen] [--no-critic] [--rounds 3] [--examples engineers_day industry_conclave]

Rare, opt-in modes for hands-on work in the VectorCraft desktop app:
    --watch [--pace 0.35] [--port 7979]   draw in the live app window so you can watch; the finished poster is left open for editing
    --finalize runs/<name> [--no-ocr]      after you edit and save it, re-export and re-run the checks on what you ended up with
"""
from __future__ import annotations

import argparse
import asyncio
import re
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
    res = asyncio.run(polish(target, default_vlm("gemini"), use_ocr=use_ocr))  # polish needs a model that returns JSON ops
    print("POLISH:", res.reason)
    if res.rationale:
        print("  rationale:", res.rationale)
    for op in res.ops_applied:
        print("  applied:", op)
    for r in res.rejected:
        print("  rejected:", r)
    if res.accepted:
        print("  polished poster ->", res.png)


def brief_from_prompt(text: str, brand: str, fmt: str | None, name: str | None) -> Path:
    """Free text -> briefs/<name>/brief.yaml (text copied verbatim from the prompt; anything not found verbatim is dropped and listed)."""
    import yaml
    from poster_agent.ingest.brief import extract_brief
    brief, problems = extract_brief(text, default_llm(), brand, fmt)
    for pr in problems:
        print("NOTE:", pr)
    base = name or re.sub(r"[^a-z0-9]+", "_", brief.event_title.lower()).strip("_")[:40] or "poster"
    folder, n = ROOT / "briefs" / base, 2
    while folder.exists() and not name:  # an explicit --name reuses its folder; a derived one never overwrites an earlier brief
        folder, n = ROOT / "briefs" / f"{base}_{n}", n + 1
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "brief.yaml").write_text(yaml.safe_dump(brief.model_dump(mode="json", exclude_none=True, exclude_defaults=True),
                                                       sort_keys=False, allow_unicode=True), encoding="utf-8")
    print("brief written ->", folder / "brief.yaml")
    return folder


def run_variants(folder: Path, args) -> None:
    """N full runs of one brief, each with a different composition recipe forced; prints the ranking and publishes the passing ones."""
    from poster_agent.outputs import publish
    from poster_agent.variants import pick_looks, pick_recipes, rank
    brief = load_brief(folder / "brief.yaml")
    brand = load_brand(brief.brand)
    inventory = scan_assets(folder / "assets") if (folder / "assets").exists() else {}
    comfy = None if args.no_gen else ComfyClient()
    llm, vlm = default_llm(), (None if args.no_critic else default_vlm(args.critic))
    runs = []
    looks = pick_looks(brief, args.variants)
    for i, recipe in enumerate(pick_recipes(brief, args.variants), 1):
        b = brief.model_copy(update={"layout_style": recipe, "look": looks[(i - 1) % len(looks)]})
        out = ROOT / "runs" / f"{folder.name}_v{i}"
        res = make_poster(b, brand, inventory, llm, out, comfy=comfy, examples=example_plans(args.examples), name=f"{folder.name}_v{i}",
                          vlm=vlm, max_refine=args.rounds)
        runs.append((recipe, res))
        print(f"variant {i} ({recipe}, look {looks[(i - 1) % len(looks)]}): {'produced' if res.ok else 'FAILED'}, passed={res.passed}", flush=True)
    ranked = rank([r for _, r in runs])
    names = {id(r): rc for rc, r in runs}
    print("ranking (best first):")
    for k, r in enumerate(ranked, 1):
        print(f"  {k}. {r.run_dir.name} ({names[id(r)]}): {'passed checks' if r.passed else 'did not pass'}" + (" <- best" if k == 1 else ""))
        if r.ok and r.passed and not args.no_publish:
            publish(r.run_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", help="free-text requirement (or @file.txt); the brief is extracted from it and saved to briefs/<name>/")
    ap.add_argument("--brand", default="kct", help="with --prompt: brand profile")
    ap.add_argument("--format", dest="fmt", default=None, help="with --prompt: story | fourfive | square")
    ap.add_argument("--name", help="with --prompt: folder/run name (default: from the event title)")
    ap.add_argument("folder", type=Path, nargs="?", help="brief folder; with --finalize: the run folder (runs/<name>) or its .vectorcraft file")
    ap.add_argument("--no-gen", action="store_true", help="do not use ComfyUI (no generated art)")
    ap.add_argument("--no-critic", action="store_true", help="skip the vision critic")
    ap.add_argument("--critic", choices=["auto", "lfm", "gemini"], default="auto",
                    help="vision model for the critic: auto = local LFM3-VL-2B (glance test) when reachable, else Gemini")
    ap.add_argument("--rounds", type=int, default=3, help="max refine rounds after the first render")
    ap.add_argument("--examples", nargs="*", default=["engineers_day"])
    ap.add_argument("--watch", action="store_true", help="draw in the live VectorCraft app window and leave the result open for editing")
    ap.add_argument("--pace", type=float, default=0.35, help="seconds between visible steps in --watch mode")
    ap.add_argument("--port", type=int, default=7979, help="VectorCraft control port for --watch")
    ap.add_argument("--finalize", action="store_true", help="re-export and re-check a poster you edited by hand")
    ap.add_argument("--no-ocr", action="store_true", help="with --finalize: skip the (slower) OCR checks")
    ap.add_argument("--polish", action="store_true", help="after the run, let the vision LLM propose a few allow-listed fine touches (kept only if checks do not worsen)")
    ap.add_argument("--variants", type=int, default=1, help="make N differently structured designs of the brief, ranked by the checks")
    ap.add_argument("--no-publish", action="store_true", help="do not copy the final image to outputs/ (evaluation and experiments)")
    ap.add_argument("--publish", action="store_true", help="copy the final image of an existing run folder (runs/<name>) to outputs/")
    ap.add_argument("--polish-only", action="store_true", help="polish an existing run folder (runs/<name>) without regenerating")
    args = ap.parse_args()

    if args.prompt:
        text = args.prompt
        if text.startswith("@"):
            text = Path(text[1:]).read_text(encoding="utf-8")
        args.folder = brief_from_prompt(text, args.brand, args.fmt, args.name)
    elif args.folder is None:
        ap.error("give a brief folder or --prompt")

    if args.finalize:
        run_finalize(args.folder, use_ocr=not args.no_ocr)
        return
    if args.publish:
        from poster_agent.outputs import publish
        print("final image ->", publish(args.folder, include_failed=True))  # an explicit --publish is the user's choice
        return
    if args.polish_only:
        run_polish(args.folder, use_ocr=not args.no_ocr)
        return

    folder = args.folder.resolve()
    if args.variants > 1:
        return run_variants(folder, args)
    brief = load_brief(folder / "brief.yaml")
    brand = load_brand(brief.brand)
    inventory = scan_assets(folder / "assets") if (folder / "assets").exists() else {}
    comfy = None if args.no_gen else ComfyClient()
    out = ROOT / "runs" / folder.name
    watch = WatchConfig(port=args.port, pace=args.pace) if args.watch else None
    res = make_poster(brief, brand, inventory, default_llm(), out, comfy=comfy, examples=example_plans(args.examples),
                      name=folder.name, vlm=None if args.no_critic else default_vlm(args.critic), max_refine=args.rounds, watch=watch)

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
    if res.ok and not args.no_publish:
        from poster_agent.outputs import publish
        final = publish(out)
        print("final image ->", final or f"not published to outputs/ (did not pass its checks; see {out})")
    if watch:
        print(f"\nThe poster is open in VectorCraft ({res.compiled.vectorcraft.name}). Edit it, save with Ctrl+S, then run:\n"
              f"    python run_poster.py --finalize runs/{folder.name}")


if __name__ == "__main__":
    main()
