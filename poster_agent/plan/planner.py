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
from .looks import apply_look, look_for
from .schema import PlanGeneratedAsset, PlanOutput
from .validate import component_assets, concept_errors, validate_plan, walk


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


def _sparse_errors(spec: PosterSpec, layout: LayoutResult) -> list[str]:
    """Advice for a plan whose layout will leave too much empty space on a plain ground (the render check would fail it). Plain grounds
    only: on generated art the picture itself fills the poster, and that can only be judged on the render."""
    from poster_agent.checks.emptiness import planned_emptiness
    from poster_agent.checks.runner import BAND_MAX, EMPTY_FRACTION_MAX, LARGEST_REGION_MAX
    if spec.background.kind not in ("solid", "gradient"):
        return []
    c = spec.canvas
    emp = planned_emptiness(layout.placed, c.width, c.height)
    over = []
    if emp.empty_fraction > EMPTY_FRACTION_MAX:
        over.append(f"{emp.empty_fraction:.0%} of the poster would be empty (limit {EMPTY_FRACTION_MAX:.0%})")
    if emp.largest_fraction > LARGEST_REGION_MAX:
        where = (f" around y {emp.regions[0][0].y / c.height:.0%}-{emp.regions[0][0].bottom / c.height:.0%} of the height"
                 if emp.regions else "")
        over.append(f"one empty region would cover {emp.largest_fraction:.0%} of the poster{where} (limit {LARGEST_REGION_MAX:.0%})")
    if emp.longest_band_fraction > BAND_MAX:
        over.append(f"an empty horizontal band would be {emp.longest_band_fraction:.0%} of the poster height (limit {BAND_MAX:.1%})")
    if not over:
        return []
    return ["layout: dead_space: " + "; ".join(over) + " — fill it with SCALE and STRUCTURE, never decoration: a larger headline "
            "(BigTitle, or title_factor up), a big HeroImage (height_pct 40 or more) or PhotoFade / ImageBlock when an asset exists, "
            "IconRow or Columns for list lines, an InfoBar band; keep all brief text"]


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
    if bg.panel is not None and background.kind in ("solid", "gradient"):
        background.panel = {"color": resolve_color(brand, bg.panel.color), "starts_at": bg.panel.starts_at}
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
        decorations=decorations, align=plan.align, flow_valign=plan.flow_valign, template=plan.template, type_style=plan.type_style,
        brief_id=_slug(brief.event_title), brand=brand.key,
        canvas=CanvasSpec.from_format(plan.format, brand.safe_margin_pct),
        palette=dict(brand.palette), fonts={k: v.family for k, v in brand.fonts.items()},
        background=background, components=components,
        assets={a: (generated[a] if a in generated else inventory[a].to_asset()) for a in sorted(used)},
        mandatory=mandatory,
    )


def _compact(plan: PlanOutput) -> tuple[PlanOutput | None, list[str]]:
    """Make a plan more compact WITHOUT touching any text: drop label-less ornament dividers, cap image / title heights, tighten
    panel padding. Returns (compacted copy, what changed) or (None, []) when nothing could be tightened."""
    out = plan.model_copy(deep=True)
    what: list[str] = []
    caps = {"HeroImage": ("height_pct", 22.0), "PhotoFade": ("height_pct", 22.0), "PhotoStrip": ("height_pct", 16.0),
            "BigTitle": ("cap_pct", 9.0), "Panel": ("padding_pct", 3.0)}

    def tighten(items: list) -> list:
        kept = []
        for c in items:
            if c.component == "Divider" and not c.content.get("label"):
                what.append(f"dropped divider {c.id}")
                continue
            key = caps.get(c.component)
            if key and float(c.content.get(key[0], 99.0)) > key[1]:
                c.content[key[0]] = key[1]
                what.append(f"{c.id}.{key[0]} -> {key[1]:g}")
            c.children = tighten(c.children)
            kept.append(c)
        return kept
    out.components = tighten(out.components)
    return (out, what) if what else (None, [])


def plan_poster(brief: Brief, brand: BrandProfile, inventory: dict[str, InventoryItem], llm: LLM, *,
                examples: list[dict] | None = None, max_rounds: int = 3, measurer: TextMeasurer | None = None,
                realize: Realize | None = None, previous: dict | None = None, problems: list[str] | None = None) -> PlanResult:
    rounds: list[RoundLog] = []
    best: PlanResult | None = None
    sparse_ok: PlanResult | None = None  # the first plan that was legal but sparse: used if the planner cannot do better

    for r in range(1, max_rounds + 1):
        try:
            raw = llm.generate_json(SYSTEM, build_user_prompt(brief, brand, inventory, examples, previous, problems))
        except LLMError as e:
            rounds.append(RoundLog(r, "llm", [str(e)[:300]]))
            if sparse_ok is not None:
                sparse_ok.rounds = rounds
                return sparse_ok
            return PlanResult(False, None, previous, None, rounds, [str(e)])
        used_llm = getattr(llm, "last_used", None) or getattr(llm, "name", None)
        try:
            plan = PlanOutput.model_validate(raw)
            look_notes = apply_look(plan, brand, look_for(brief))  # ground, type style, title treatment, colours, one image idea
            plan.rationale = (plan.rationale + " [look] " + " | ".join(look_notes))[:900]
            raw = plan.model_dump(mode="json")
        except ValidationError as e:
            problems = [f"schema: {'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()][:12]
            rounds.append(RoundLog(r, "schema", problems, used_llm))
            previous = raw
            continue

        errs = validate_plan(plan, brief, brand, inventory) + concept_errors(plan)
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
            soft = _sparse_errors(spec, layout) if sparse_ok is None else []  # one chance to improve a sparse plan
            if soft and r < max_rounds:
                sparse_ok = PlanResult(True, spec, raw, layout, rounds, output=plan)
                rounds.append(RoundLog(r, "layout", soft, used_llm))
                problems, previous = soft, raw
                continue
            rounds.append(RoundLog(r, "ok", [], used_llm))
            return PlanResult(True, spec, raw, layout, rounds, output=plan)
        compact, what = _compact(plan)  # a plan that only fails on height is shrunk deterministically before another LLM round
        if compact is not None:
            spec2 = build_spec(compact, brief, brand, inventory, generated)
            layout2 = apply_layout(spec2, brand, measurer)
            if not _layout_errors(layout2):
                rounds.append(RoundLog(r, "ok", ["compacted automatically (content unchanged): " + "; ".join(what)], used_llm))
                return PlanResult(True, spec2, compact.model_dump(mode="json"), layout2, rounds, output=compact)
        rounds.append(RoundLog(r, "layout", layout_errs, used_llm))
        problems, previous = layout_errs, raw
        if best is None or len(layout_errs) < len(best.problems):
            best = PlanResult(False, spec, raw, layout, rounds, layout_errs, output=plan)

    if sparse_ok is not None:  # the planner's attempts to fill the space failed; the legal sparse plan is still a valid poster
        sparse_ok.rounds = rounds
        return sparse_ok
    if best is not None:
        best.rounds = rounds
        return best
    return PlanResult(False, None, previous, None, rounds, problems or ["no plan produced"])
