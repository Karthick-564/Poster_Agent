"""Issue router: every typed issue goes to the stage that owns the fix. Pure and deterministic — the table is the policy.

  repair   deterministic spec edits (e.g. recolour text)        low_contrast
  assets   regenerate generated art with a new seed             bg_clash, hero_quality, stray_text (in art)
  plan     replan with the issues as feedback                   hierarchy, density, component_choice, spacing, alignment,
                                                                 fit, text_overlap, margin_violation, mandatory_missing, logo_violation
  compile  re-render once (a render problem, not a design one)  render_artifact, output_spec

A repair that did not work escalates (to assets if the art is generated, else to plan) instead of being retried forever.
"""
from __future__ import annotations

from poster_agent.spec import AssetKind, Issue, IssueCategory, PosterSpec

STAGE_BY_CATEGORY: dict[IssueCategory, str] = {
    IssueCategory.low_contrast: "repair",
    IssueCategory.dead_space: "enrich",
    IssueCategory.bg_clash: "assets",
    IssueCategory.hero_quality: "assets",
    IssueCategory.stray_text: "assets",
    IssueCategory.photo_treatment: "assets",
    IssueCategory.render_artifact: "compile",
    IssueCategory.output_spec: "compile",
}
GENERATED = {AssetKind.generated_background, AssetKind.generated_hero}


def has_generated_art(spec: PosterSpec) -> bool:
    return spec.background.kind == "generated" or any(a.kind in GENERATED for a in spec.assets.values())


def _generated_ids(spec: PosterSpec) -> set[str]:
    """Ids an issue may cite to point at generated art: the assets, the elements showing them, and the background."""
    assets = {a.id for a in spec.assets.values() if a.kind in GENERATED}
    elements = {e.id for e in spec.placed if e.asset_id in assets}
    return assets | elements | ({"background"} if spec.background.kind == "generated" else set())


def implicated_assets(issues: list[Issue], spec: PosterSpec) -> set[str]:
    """Generated assets (and \"background\") the issues point at, so only those are regenerated."""
    heroes = {a.id for a in spec.assets.values() if a.kind == AssetKind.generated_hero}
    by_element = {e.id: e.asset_id for e in spec.placed if e.asset_id}
    out: set[str] = set()
    for i in issues:
        for eid in i.element_ids:
            if eid == "background" or eid in spec.assets:
                out.add(eid)
            elif eid in by_element:
                out.add(by_element[eid])
        if not i.element_ids and i.category == IssueCategory.hero_quality:
            out |= heroes
        if not i.element_ids and i.category == IssueCategory.bg_clash and spec.background.kind == "generated":
            out.add("background")
    return out


def route_issue(issue: Issue, spec: PosterSpec, attempts: dict[str, int]) -> str:
    stage = STAGE_BY_CATEGORY.get(issue.category, "plan")
    art = has_generated_art(spec)
    if stage == "assets" and not art:
        return "plan"  # nothing generated to regenerate: the planner must change the design
    if issue.category in (IssueCategory.stray_text, IssueCategory.photo_treatment):
        mine = _generated_ids(spec)
        if issue.element_ids:
            if not any(i in mine for i in issue.element_ids):
                return "plan"  # it is about something we did not generate (a supplied photo, a text element)
        elif not (issue.category == IssueCategory.stray_text and spec.background.kind == "generated"):
            return "plan"
    if stage == "enrich" and attempts.get("enrich", 0) >= 2:
        return "plan"  # decoration did not close the gap: the design needs more content or a bigger hero
    if stage == "repair" and attempts.get("repair", 0) >= 1:
        return "assets" if art else "plan"  # recolouring already tried and it is still failing
    if stage == "compile" and attempts.get("compile", 0) >= 1:
        return "halt"  # re-rendering did not help: a real defect, to be reported
    return stage


def route(issues: list[Issue], spec: PosterSpec, attempts: dict[str, int] | None = None) -> dict[str, list[Issue]]:
    attempts = attempts or {}
    out: dict[str, list[Issue]] = {}
    for i in issues:
        out.setdefault(route_issue(i, spec, attempts), []).append(i)
    return out


def feedback_lines(issues: list[Issue]) -> list[str]:
    """Plain-English problems for the planner prompt."""
    return [f"{i.source} {i.category.value}: {i.evidence}" + (f" — fix: {i.suggested_fix}" if i.suggested_fix else "")
            for i in issues]
