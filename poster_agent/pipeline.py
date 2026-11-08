"""End-to-end run with a bounded refine loop.

    plan -> [ render -> pixel checks -> vision critic -> route issues -> repair / regenerate art / replan ] x up to N rounds -> best

Honest by construction: planning failure halts with the problems listed; failed art generation falls back to the solid ground
and says so; an unavailable critic is reported as unavailable (never a made-up score); the best-scoring round is kept and the
final report says whether it actually passed.
"""
from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path

from PIL import Image

from poster_agent.analysis import CALM_DETAIL, analyse, analyse_image, text_detail
from poster_agent.assets import ArtDirection, AssetGenerator, ComfyClient, GenInfo, calm_layout
from poster_agent.assets.cutouts import apply_cutouts
from poster_agent.assets.person import apply_framing
from poster_agent.assets.treatments import apply_treatments, calm_under_text
from poster_agent.checks import CheckReport, run_checks
from poster_agent.compile.compiler import CompileResult, compile_spec
from poster_agent.compile.vc_client import VectorCraftSession, WatchConfig
from poster_agent.critic import CriticReport, critique
from poster_agent.enrich import default_preset, hero_glow, is_dark_ground, top_up
from poster_agent.ingest import Brief, InventoryItem
from poster_agent.layout.adapt import adapt_to_background
from poster_agent.layout.base import resolve_color
from poster_agent.layout.candidates import apply_choice, choose_layout
from poster_agent.layout.solver import apply_layout, solve_layout
from poster_agent.llm import LLM
from poster_agent.plan import PlanResult, plan_poster
from poster_agent.plan.schema import PlanGeneratedAsset, PlanOutput
from poster_agent.repairs import repair_contrast
from poster_agent.router import feedback_lines, has_generated_art, implicated_assets, route
from poster_agent.spec import AssetKind, AssetRef, BackgroundSpec, Box, ElementKind, Issue, IssueCategory, PosterSpec
from poster_agent.spec.brand import ROOT, BrandProfile


@dataclass
class RoundRecord:
    n: int
    dir: Path
    compiled: CompileResult
    spec_json: str
    checks: CheckReport | None
    critic: CriticReport | None
    errors: int
    actionable: list[Issue] = field(default_factory=list)  # check errors + critic issues found in this round
    actions: list[str] = field(default_factory=list)  # what the loop did AFTER this round

    @property
    def score(self) -> int | None:
        return self.critic.score if self.critic and self.critic.available else None

    @property
    def empty_fraction(self) -> float:
        emp = next((r for r in (self.checks.results if self.checks else []) if r.name == "emptiness"), None)
        return float(emp.measured.get("empty_fraction", 0.0)) if emp else 0.0

    @property
    def key(self) -> tuple[int, int, float]:
        """Lower is better: fewer errors, then a higher critic score, then LESS dead space (so enrichment rounds that
        reduce emptiness count as improvements even when the error count is unchanged)."""
        return (self.errors, -(self.score or 0), round(self.empty_fraction, 3))

    @property
    def passed(self) -> bool:
        return (self.checks is None or self.checks.ok) and (self.critic is None or not self.critic.available or self.critic.passed)


@dataclass
class RunResult:
    ok: bool  # a poster was rendered
    run_dir: Path
    plan: PlanResult
    spec: PosterSpec | None = None
    compiled: CompileResult | None = None
    generated: list[GenInfo] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    checks: CheckReport | None = None
    critic: CriticReport | None = None
    rounds: list[RoundRecord] = field(default_factory=list)
    best_round: int | None = None
    passed: bool = False  # the kept poster has no failing checks and (if reviewed) a passing critic score

    @property
    def reviewed(self) -> bool:
        """True only if the vision critic actually looked at the kept poster."""
        return bool(self.critic and self.critic.available)


def _report(r: RunResult) -> dict:
    return {
        "ok": r.ok, "passed": r.passed, "reviewed": r.reviewed, "best_round": r.best_round,
        "plan_rounds": [{"round": x.round, "stage": x.stage, "llm": x.llm, "problems": x.problems} for x in r.plan.rounds],
        "problems": r.plan.problems,
        "generated": [g.__dict__ for g in r.generated],
        "notes": r.notes,
        "checks": r.checks.to_dict() if r.checks else None,
        "critic": r.critic.to_dict() if r.critic else None,
        "refine_rounds": [{"round": x.n, "errors": x.errors, "critic_score": x.score, "passed": x.passed,
                           "issues": [{"source": i.source, "category": i.category.value, "severity": i.severity,
                                       "evidence": i.evidence[:240]} for i in x.actionable],
                           "actions_after": x.actions, "dir": str(x.dir)} for x in r.rounds],
        "issues": [i.model_dump(mode="json") for i in (r.spec.issues if r.spec else [])],
        "outputs": ({"png": str(r.compiled.png), "pdf": str(r.compiled.pdf), "vectorcraft": str(r.compiled.vectorcraft),
                     "overflow": r.compiled.overflow, "warnings": r.compiled.warnings} if r.compiled else None),
    }


def _write_report(r: RunResult) -> None:
    (r.run_dir / "report.json").write_text(json.dumps(_report(r), indent=1, ensure_ascii=False), encoding="utf-8")


BACKGROUND_CANDIDATES = 5  # at most this many pictures per background; generation stops early at the first one calm enough under the text
DETAIL_WEIGHT = 20.0  # placement cost + this x the fine detail under the text decides between pictures


def _realize_background(spec: PosterSpec, plan: PlanOutput, gen: AssetGenerator | None, brand: BrandProfile, offset: int,
                        infos: list[GenInfo], notes: list[str]) -> None:
    bg = plan.background
    if bg.kind != "generated" or bg.art is None:
        return
    if gen is None:
        msg = "background art was requested but no ComfyUI client is configured; solid ground used"
        if msg not in notes:
            notes.append(msg)
        return
    art = ArtDirection(description=bg.art.description, style=bg.art.style, lighting=bg.art.lighting, medium=bg.art.medium,
                       tone=bg.art.tone, palette=[resolve_color(brand, c) for c in bg.art.palette], subject=bg.art.subject,
                       composition=bg.art.composition, devices=bg.art.devices, color_roles=bg.art.color_roles)
    calm = calm_layout(spec.placed, spec.canvas.width, spec.canvas.height)  # text first: the art prompt says where the quiet space must be
    try:
        best: tuple[float, AssetRef, GenInfo, int] | None = None
        costs: list[float] = []
        details: list[float] = []
        for k in range(BACKGROUND_CANDIDATES):  # then several pictures; keep the one the text fits best (measured, not judged by eye)
            ref_k, info_k = gen.background(f"background_{k}", art, spec.canvas, calm, seed_offset=offset + 101 * k)
            infos.append(info_k)
            with Image.open(ROOT / ref_k.path) as im:
                art_map = analyse_image(im, spec.canvas.width, spec.canvas.height)
            ch = choose_layout(spec, brand, art_map)
            cost, detail = (ch.chosen.cost if ch is not None else 0.0), 0.0
            if ch is not None:  # how detailed is this picture under the text, once the text has moved to its best place on it?
                sp = spec.model_copy(update={"flow_region": ch.chosen.region, "flow_valign": ch.chosen.valign})
                boxes = [e.box for e in solve_layout(sp, brand).placed if e.kind == ElementKind.text and not e.props.get("decorative")]
                with Image.open(ROOT / ref_k.path) as im:
                    detail = text_detail(im, boxes, spec.canvas.width, spec.canvas.height)
            cost += DETAIL_WEIGHT * detail
            details.append(detail)
            costs.append(cost)
            if best is None or cost < best[0]:
                best = (cost, ref_k, info_k, k)
            if detail <= CALM_DETAIL:
                break  # calm enough under the text: no need to spend more GPU time
        _, ref, info, k_won = best
        ref = ref.model_copy(update={"id": "background"})  # the kept picture keeps the id the rest of the pipeline refers to
        notes.append(f"background: kept candidate {k_won} of {len(costs)} generated (max {BACKGROUND_CANDIDATES}); placement cost + detail under "
                     f"the text: {', '.join(f'{c:.2f}' for c in costs)}; detail under the text: {', '.join(f'{d:.3f}' for d in details)}")
        spec.assets[ref.id] = ref
        spec.background = BackgroundSpec(kind="generated", asset_id=ref.id, color=spec.background.color)
    except Exception as e:  # honest fallback: keep the solid ground and record it
        msg = f"generated background failed ({str(e)[:200]}); solid ground used instead"
        notes.append(msg)
        spec.issues.append(Issue(category=IssueCategory.bg_clash, severity="warning", evidence=msg, source="check"))


def _calm_under_text(spec: PosterSpec, brand: BrandProfile, tone: str, rdir: Path, notes: list[str]) -> None:
    """After the text is placed: if the picture is still detailed under it, soften only those areas (feathered blur + brand-tint wash).
    The untouched picture stays on disk; the spec points at the softened copy. Says what was measured before and after."""
    ref = spec.assets.get(spec.background.asset_id or "")
    if spec.background.kind != "generated" or ref is None:
        return
    boxes = [e.box for e in spec.placed if e.kind == ElementKind.text and not e.props.get("decorative")]
    if not boxes:
        return
    with Image.open(ROOT / ref.path) as im:
        out, info = calm_under_text(im, boxes, spec.canvas.width, spec.canvas.height,
                                    resolve_color(brand, "primary" if tone == "dark" else "surface"), CALM_DETAIL)
    if info["passes"] == 0:
        msg = f"art under the text is already calm (detail {info['before']:.3f}, target {info['target']:.3f}); left as generated"
    else:
        dest = rdir / "assets" / "background_calm.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.save(dest)
        spec.assets[ref.id] = ref.model_copy(update={"path": dest.resolve().relative_to(ROOT).as_posix(),
                                                      "treatment": {**ref.treatment, "calm_under_text": info, "source": ref.path}})
        msg = (f"art softened under the text only ({info['passes']} pass(es)): detail {info['before']:.3f} -> {info['after']:.3f} "
               f"(target {info['target']:.3f}); the rest of the picture is untouched")
    if msg not in notes:
        notes.append(msg)


async def make_poster_async(brief: Brief, brand: BrandProfile, inventory: dict[str, InventoryItem], llm: LLM, run_dir: Path, *,
                            comfy: ComfyClient | None = None, examples: list[dict] | None = None, name: str = "poster",
                            cache_dir: Path | None = None, checks: bool = True, vlm: LLM | None = None,
                            max_refine: int = 3, decorate: bool = True, check_emptiness: bool = True,
                            watch: WatchConfig | None = None) -> RunResult:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    gen = AssetGenerator(comfy, run_dir / "assets", cache_dir) if comfy else None
    infos: list[GenInfo] = []
    hero_offset = [0]

    def realize(items: list[PlanGeneratedAsset]) -> dict[str, AssetRef]:
        out: dict[str, AssetRef] = {}
        for it in items:
            ref, info = gen.hero_cutout(it.id, it.subject, aspect=it.aspect, seed_offset=hero_offset[0])
            infos.append(info)
            out[it.id] = ref
        return out

    pr = plan_poster(brief, brand, inventory, llm, examples=examples, realize=realize if gen else None)
    result = RunResult(ok=False, run_dir=run_dir, plan=pr, spec=pr.spec, generated=infos)
    if pr.plan is not None:
        (run_dir / "plan.json").write_text(json.dumps(pr.plan, indent=1, ensure_ascii=False), encoding="utf-8")
    if not pr.ok or pr.spec is None or pr.output is None:
        result.notes.append("planning failed; nothing was rendered")
        _write_report(result)
        return result

    def adopt(sp: PosterSpec, po: PlanOutput) -> None:
        """A plain background with no planned decoration gets the brand's default decorative layer (never plain by default)."""
        if decorate and not sp.decorations and sp.background.kind in ("solid", "gradient") and po.background.kind != "generated":
            sp.decorations = default_preset(sp, brand, is_dark_ground(sp, brand))
            apply_layout(sp, brand)
        elif decorate and not sp.decorations and po.background.kind == "generated":
            sp.decorations = default_preset(sp, brand, is_dark_ground(sp, brand, po.background.art.tone if po.background.art else None))
            apply_layout(sp, brand)
        if decorate and sp.decorations and not any(d.kind == "glow" for d in sp.decorations):
            # a hero always gets its setting, even when the planner chose its own (restrained) decorations
            g = hero_glow(sp, brand, is_dark_ground(sp, brand, po.background.art.tone if po.background.art else None))
            if g is not None:
                sp.decorations.append(g)
                apply_layout(sp, brand)

    spec, plan_out, plan_dict = pr.spec, pr.output, pr.plan
    adopt(spec, plan_out)
    attempts: dict[str, int] = {}
    bg_offset = 0
    best: RoundRecord | None = None
    best_plan: tuple[PlanOutput, dict] = (plan_out, plan_dict)
    stall = 0

    async with VectorCraftSession(watch=watch) as vc:
        for rnd in range(max_refine + 1):
            rdir = run_dir / f"round_{rnd}"
            _realize_background(spec, plan_out, gen, brand, bg_offset, infos, result.notes)
            result.notes += [n for n in apply_framing(spec, rdir / "assets") if n not in result.notes]  # crop people photos before treatment
            result.notes += [n for n in apply_cutouts(spec, rdir / "assets") if n not in result.notes]  # then matte supplied cutout photos
            result.notes += [n for n in apply_treatments(spec, brand, rdir / "assets") if n not in result.notes]
            art_map = analyse(spec, ROOT)  # look at the art first, then decide where the text goes
            choice = choose_layout(spec, brand, art_map) if art_map is not None else None  # every template: the text follows the art
            if choice is not None:
                apply_choice(spec, choice)
                apply_layout(spec, brand)
                if choice.note() not in result.notes:
                    result.notes.append(choice.note())
            if plan_out.background.kind == "generated" and plan_out.background.art is not None:
                _calm_under_text(spec, brand, plan_out.background.art.tone, rdir, result.notes)  # last resort: soften the art under the text only
            result.notes += [f"text adapted to the background: {n}" for n in adapt_to_background(spec, brand, ROOT)
                             if f"text adapted to the background: {n}" not in result.notes]
            compiled = await compile_spec(spec, vc, rdir, name=name, watched=watch is not None,
                                          pace=watch.pace if watch else 0.0, doc_title=f"{name} - round {rnd}")
            chk = run_checks(spec, brand, compiled, use_emptiness=check_emptiness) if checks else None
            crit = critique(compiled.png, spec, brand, brief, chk, vlm, rdir) if vlm else None
            n_err = (sum(i.severity == "error" for i in chk.issues) if chk else 0) + \
                    (sum(i.severity == "error" for i in crit.issues) if crit and crit.available else 0)
            (rdir / "spec.json").write_text(spec.model_dump_json(indent=1), encoding="utf-8")
            actionable_now = ([i for i in chk.issues if i.severity == "error"] if chk else []) + \
                             (crit.issues if crit and crit.available else [])
            rec = RoundRecord(rnd, rdir, compiled, spec.model_dump_json(), chk, crit, n_err, actionable_now)
            result.rounds.append(rec)

            regress: list[str] = []
            actions: list[str] = []
            if best is None or rec.key < best.key:
                best, best_plan, stall = rec, (plan_out, plan_dict), 0
            else:
                # This revision is not better. Go back to the best design instead of piling more changes on a worse one,
                # and tell the planner what the failed attempt broke.
                stall += 1
                new = [i.evidence[:140] for i in rec.actionable if i.severity == "error"][:3]
                regress = ["a previous revision of this design made it worse" + (" and introduced: " + " | ".join(new) if new else "")
                           + ". Do not repeat that change; make the smallest change that fixes the problems below."]
                spec = PosterSpec.model_validate_json(best.spec_json)
                plan_out, plan_dict = best_plan
                actions.append(f"round {rnd} was not better than round {best.n}: reverted to round {best.n}")
                result.notes.append(f"round {rnd} (errors={rec.errors}, score={rec.score}) was not better than round {best.n}; reverted")
            if rec.passed or rnd == max_refine:
                rec.actions = actions
                break
            if stall >= 2:
                result.notes.append(f"stopped after round {rnd}: two revisions in a row did not improve on round {best.n}")
                rec.actions = actions
                break

            routed = route(best.actionable, spec, attempts)

            repair_issues = routed.pop("repair", [])
            if repair_issues:
                acts, unresolved = repair_contrast(spec, brand, best.compiled, repair_issues)
                attempts["repair"] = attempts.get("repair", 0) + 1
                actions += acts
                if acts:
                    apply_layout(spec, brand)  # re-applies overrides to the rebuilt placed elements
                for i in unresolved:
                    routed.setdefault("assets" if has_generated_art(spec) else "plan", []).append(i)
                if acts:
                    # Cheap deterministic fixes first. Unreadable text also shows up as OCR 'missing text', so re-render and
                    # re-check before spending regeneration or an LLM replan on what may only be a symptom.
                    rec.actions = actions
                    continue
            enrich_issues = routed.pop("enrich", [])
            if enrich_issues:
                emp = next((r for r in (best.checks.results if best.checks else []) if r.name == "emptiness"), None)
                regs = [Box(x=r["x"], y=r["y"], w=r["w"], h=r["h"]) for r in (emp.measured.get("regions", []) if emp else [])]
                tone = plan_out.background.art.tone if plan_out.background.art else None
                acts = top_up(spec, brand, regs, is_dark_ground(spec, brand, tone), attempts.get("enrich", 0))
                attempts["enrich"] = attempts.get("enrich", 0) + 1
                actions += acts
                if acts:
                    apply_layout(spec, brand)
                    rec.actions = actions
                    continue
                routed.setdefault("plan", []).extend(enrich_issues)
            if routed.pop("compile", []):
                attempts["compile"] = attempts.get("compile", 0) + 1
                actions.append("re-render (render/output issue)")
            routed.pop("halt", None)  # defects a re-render did not fix stay in the report

            asset_issues = routed.pop("assets", [])
            if asset_issues and gen and has_generated_art(spec):
                hit = implicated_assets(asset_issues, spec)  # regenerate only what the issues point at
                hero_hit = any(a in spec.assets and spec.assets[a].kind == AssetKind.generated_hero for a in hit)
                if hero_hit and plan_out.generated_assets:
                    hero_offset[0] += 1
                    try:
                        for aid, ref in realize(plan_out.generated_assets).items():
                            spec.assets[aid] = ref
                        apply_layout(spec, brand)
                        actions.append("regenerated hero cutout(s) with a new seed")
                    except Exception as e:
                        actions.append(f"hero regeneration failed: {str(e)[:120]}")
                if spec.background.kind == "generated" and ("background" in hit or not hit):
                    bg_offset += 1
                    actions.append("regenerating the background with a new seed")
                if not any(a.startswith(("regenerat", "hero regeneration failed")) for a in actions):
                    routed.setdefault("plan", []).extend(asset_issues)  # nothing to regenerate: the design must change
            elif asset_issues:
                routed.setdefault("plan", []).extend(asset_issues)

            plan_issues = routed.pop("plan", [])
            if plan_issues:
                pr2 = plan_poster(brief, brand, inventory, llm, examples=examples, realize=realize if gen else None,
                                  previous=plan_dict, problems=feedback_lines(plan_issues) + regress, max_rounds=2)
                if pr2.ok and pr2.spec is not None and pr2.output is not None:
                    spec, plan_out, plan_dict = pr2.spec, pr2.output, pr2.plan
                    adopt(spec, plan_out)
                    actions.append(f"replanned from {len(plan_issues)} issue(s)")
                else:
                    actions.append("replan failed; kept the current design: " + "; ".join(pr2.problems)[:160])
            rec.actions = actions
            if not actions or all(a.startswith(("replan failed", "hero regeneration failed", "round ")) for a in actions):
                result.notes.append(f"stopped after round {rnd}: no further automatic fix available")
                break

    best = min(result.rounds, key=lambda r: r.key)
    result.best_round = best.n
    (run_dir / "plan.json").write_text(json.dumps(best_plan[1], indent=1, ensure_ascii=False), encoding="utf-8")  # the plan of the KEPT poster
    result.spec = PosterSpec.model_validate_json(best.spec_json)
    result.checks, result.critic = best.checks, best.critic
    result.passed = best.passed
    result.ok = True
    top: dict[str, Path] = {}
    for attr, ext in (("png", ".png"), ("pdf", ".pdf"), ("vectorcraft", ".vectorcraft")):
        dst = run_dir / f"{name}{ext}"
        shutil.copyfile(Path(getattr(best.compiled, attr)), dst)
        top[attr] = dst
    result.compiled = replace(best.compiled, **top)
    (run_dir / "spec.json").write_text(best.spec_json, encoding="utf-8")
    _write_report(result)
    if watch:  # leave the kept poster open in the app: your Ctrl+S overwrites exactly the file `--finalize` reads
        async with VectorCraftSession(watch=watch) as vc:
            await vc.command("file.open", path=str(top["vectorcraft"]))
    return result


def make_poster(*args, **kwargs) -> RunResult:
    return asyncio.run(make_poster_async(*args, **kwargs))
