"""plan_poster: brief + brand + inventory -> validated PosterSpec with layout applied.

Loop: LLM plan -> schema check -> deterministic validation -> layout solve; any problem (including the solver's
`fit` issues) goes back to the LLM as plain-English feedback. Max rounds, then an honest failure — never a silent fallback.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable
from typing import Any

from pydantic import ValidationError

from poster_agent.ingest import Brief, InventoryItem
from poster_agent.layout.base import resolve_color
from poster_agent.layout.solver import LayoutResult, apply_layout
from poster_agent.layout.fonts import TextMeasurer
from poster_agent.llm import LLM, LLMError
from PIL import Image

from poster_agent.spec import AssetKind, AssetRef, BackgroundSpec, Decoration, CanvasSpec, GradientStop, MandatoryItem, PosterSpec
from poster_agent.spec.brand import ROOT, BrandProfile

from .prompt import SYSTEM, build_user_prompt
from .schema import PlanGeneratedAsset, PlanOutput
from .validate import component_assets, validate_plan, walk


@dataclass
class RoundLog:
    round: int
    stage: str  # llm | schema | validate | layout | ok
    problems: list[str] = field(default_factory=list)
    llm: str | None = None


@dataclass
class PlanResult:
    ok: bool
    spec: PosterSpec | None
    plan: dict | None
    layout: LayoutResult | None
    rounds: list[RoundLog]
    problems: list[str] = field(default_factory=list)  # what was still wrong at the end (empty when ok)
    output: PlanOutput | None = None  # the validated plan (the background's art direction lives here)


Realize = Callable[[list[PlanGeneratedAsset]], dict[str, AssetRef]]


def _placeholder_asset(g: PlanGeneratedAsset) -> AssetRef:
    """A transparent stand-in with the declared aspect, so layout can be tried before any GPU time is spent."""
    a, b = (int(x) for x in g.aspect.split(":"))
    path = ROOT / "data/cache/placeholders" / f"ar_{a}x{b}.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (a * 100, b * 100), (0, 0, 0, 0)).save(path)
    return AssetRef(id=g.id, kind=AssetKind.generated_hero, path=path.relative_to(ROOT).as_posix(), provenance="generated")


def _layout_errors(layout: LayoutResult) -> list[str]:
    return [f"layout: {i.category.value}: {i.evidence}" + (f" — {i.suggested_fix}" if i.suggested_fix else "")
            for i in layout.issues if i.severity == "error"]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48] or "poster"


def build_spec(plan: PlanOutput, brief: Brief, brand: BrandProfile, inventory: dict[str, InventoryItem],
               generated: dict[str, AssetRef] | None = None) -> PosterSpec:
    bg = plan.background
    generated = generated or {}
    if bg.kind == "generated" and bg.art is not None:
        # placeholder ground; the pipeline swaps in the generated art once it knows where the text sits
        ground = brand.palette["primary"] if bg.art.tone == "dark" else brand.palette["surface"]
        background = BackgroundSpec(kind="solid", color=ground)
    elif bg.kind == "gradient" and len(bg.gradient) >= 2:
        background = BackgroundSpec(kind="gradient", angle=bg.angle, gradient=[
            GradientStop(offset=s.offset, color=resolve_color(brand, s.color)) for s in bg.gradient])
    else:
        background = BackgroundSpec(kind="solid", color=resolve_color(brand, bg.color))
    components = list(plan.components)
    used: set[str] = set()
    for c in walk(components):
        ids = component_assets(c)
        c.asset_ids = ids
        used |= set(ids)
    mandatory = [MandatoryItem(id=p, label=p, text=t) for p, t in brief.required_fields()]
    mandatory.append(MandatoryItem(id="logo", kind="logo", label="Institution logo"))
    decorations = [Decoration(id=f"plan{i}_{d.kind}", kind=d.kind, region=d.region, color=d.color, opacity=d.opacity, text=d.text,
                              density=d.density, source="plan") for i, d in enumerate(plan.decorations)]
    return PosterSpec(
        decorations=decorations, align=plan.align,
        brief_id=_slug(brief.event_title), brand=brand.key,
        canvas=CanvasSpec.from_format(plan.format, brand.safe_margin_pct),
        palette=dict(brand.palette), fonts={k: v.family for k, v in brand.fonts.items()},
        background=background, components=components,
        assets={a: (generated[a] if a in generated else inventory[a].to_asset()) for a in sorted(used)},
        mandatory=mandatory,
    )


def plan_poster(brief: Brief, brand: BrandProfile, inventory: dict[str, InventoryItem], llm: LLM, *,
                examples: list[dict] | None = None, max_rounds: int = 3, measurer: TextMeasurer | None = None,
                realize: Realize | None = None, previous: dict | None = None, problems: list[str] | None = None) -> PlanResult:
    rounds: list[RoundLog] = []
    best: PlanResult | None = None

    for r in range(1, max_rounds + 1):
        try:
            raw = llm.generate_json(SYSTEM, build_user_prompt(brief, brand, inventory, examples, previous, problems))
        except LLMError as e:
            rounds.append(RoundLog(r, "llm", [str(e)[:300]]))
            return PlanResult(False, None, previous, None, rounds, [str(e)])
        used_llm = getattr(llm, "last_used", None) or getattr(llm, "name", None)
        try:
            plan = PlanOutput.model_validate(raw)
        except ValidationError as e:
            problems = [f"schema: {'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()][:12]
            rounds.append(RoundLog(r, "schema", problems, used_llm))
            previous = raw
            continue

        errs = validate_plan(plan, brief, brand, inventory)
        if errs:
            problems, previous = errs, raw
            rounds.append(RoundLog(r, "validate", errs, used_llm))
            continue

        generated: dict[str, AssetRef] = {}
        if plan.generated_assets or plan.background.kind == "generated":
            if realize is None:
                problems, previous = ["generated art (generated_assets / background.kind 'generated') is not available in "
                                      "this run; use supplied assets, solid or gradient backgrounds only"], raw
                rounds.append(RoundLog(r, "validate", problems, used_llm))
                continue
            if plan.generated_assets:
                # Dry run: lay the plan out with placeholders first; a plan that cannot fit is not worth GPU minutes.
                dry_spec = build_spec(plan, brief, brand, inventory, {g.id: _placeholder_asset(g) for g in plan.generated_assets})
                dry_errs = _layout_errors(apply_layout(dry_spec, brand, measurer))
                if dry_errs:
                    problems, previous = dry_errs, raw
                    rounds.append(RoundLog(r, "layout", dry_errs, used_llm))
                    continue
            try:
                generated = realize(plan.generated_assets) if plan.generated_assets else {}
            except Exception as e:  # generation failed: tell the planner so it can drop or change the hero
                problems, previous = [f"asset generation failed: {str(e)[:300]}"], raw
                rounds.append(RoundLog(r, "assets", problems, used_llm))
                continue
        spec = build_spec(plan, brief, brand, inventory, generated)
        layout = apply_layout(spec, brand, measurer)
        layout_errs = _layout_errors(layout)
        if not layout_errs:
            rounds.append(RoundLog(r, "ok", [], used_llm))
            return PlanResult(True, spec, raw, layout, rounds, output=plan)
        rounds.append(RoundLog(r, "layout", layout_errs, used_llm))
        problems, previous = layout_errs, raw
        if best is None or len(layout_errs) < len(best.problems):
            best = PlanResult(False, spec, raw, layout, rounds, layout_errs, output=plan)

    if best is not None:
        best.rounds = rounds
        return best
    return PlanResult(False, None, previous, None, rounds, problems or ["no plan produced"])
