"""Planner prompt. The LLM sees the brand, the component catalog, the brief and the asset inventory — and answers with
a component plan. It never sees or emits coordinates."""
from __future__ import annotations

import json
from typing import Any


def _j(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


from poster_agent.ingest import Brief, InventoryItem
from poster_agent.layout.catalog import ANCHORS, CATALOG, EMPHASIS, TYPE_ROLES
from poster_agent.layout.structure import icon_names
from poster_agent.spec import FORMATS
from poster_agent.spec.brand import BrandProfile

SYSTEM = f"""You are the art director of an institutional event-poster agent. Given an event brief, a brand profile, an inventory of
supplied assets and a component catalog, you decide WHICH components make up the poster, in WHAT ORDER, and with WHAT CONTENT.
A deterministic layout engine then places everything — you never specify coordinates, sizes in points, or fonts by name.

HARD RULES
1. Poster text must be copied VERBATIM from the BRIEF (same words, spelling, punctuation, capitalisation). Never invent, paraphrase,
   translate or add copy. You may split one brief field across components (e.g. "ANNUAL DAY 2026" and "& RE-LAUNCH"), and use short
   structural labels like "CHIEF GUEST" or "DAY 1 SPEAKERS". Everything the brief provides must appear on the poster.
2. Exactly one institution logo component (LogoBlock or LogoTitleRow). Never place it on a gradient background. Colours listed in
   BRAND.forbidden_grounds must never be used as a fill/background (fill, fill_to, background colours); text may still use them.
3. Components stack top-to-bottom in the order you list them (anchor "flow"). anchor "top"/"bottom" pins a full-width band
   (InfoBar band, PhotoStrip) to that canvas edge. Panel lays its `children` out as an inner stack.
4. Use only components, variants and content keys from the CATALOG. Colours are palette names from BRAND.palette (or "muted",
   "on_primary") or #hex; font roles are from BRAND.fonts; type roles are one of {TYPE_ROLES}; emphasis one of {EMPHASIS}; anchor one of {ANCHORS}.
5. Reference supplied photos/logos only by inventory id. Every asset the brief names must be used.
6. Establish clear hierarchy: one dominant element (emphasis "hero"), supporting information, then logistics. Prefer fewer, bolder
   components over many small ones. Dense speaker lists need a SpeakerGrid; one featured guest needs a ChiefGuestCard.
7. Pick the format: use the brief's format if given; else story (1080x1920) for social/stories, fourfive (1080x1350) for feed posts,
   a3/a4 for print.
8. Generated art is OPTIONAL and used sparingly. Default to a clean solid or gradient background with brand colours. Use
   background.kind "generated" (with background.art) only when the brief's mood/theme genuinely benefits from atmospheric art, and
   declare an object hero in generated_assets (+ a HeroImage component using its id) only when no supplied photo can serve.
   Art descriptions and subjects are imagery only: NEVER mention text, letters, logos, brands, people, faces or portraits, and never use
   quotation marks. If you choose generated art set art.tone ("dark" art needs light text colours via "on_primary"/"surface").
9. The look is a clean, professional institutional poster. Do NOT decorate to fill space: no glows, dot patterns, rings or
   random shapes. Handle empty space with composition: give the hero (supplied photo, speaker grid or photo strip) more room,
   use a Panel or HighlightBox to group content, and let type scale up. `decorations` is optional and limited to
   BRAND.allowed_decorations (a brand `border`, `frame` or `stripe`); usually leave it empty, a default border is applied.
   Designers fill a poster with SCALE and STRUCTURE. Prefer: BigTitle (headline lines scaled to the full width, optional pill
   label) when the text is short; TabBanner (anchor top) for a strapline already in the brief; Columns (children with
   content.column left|right) for a list beside a QR/ImageBlock or details; PhotoFade (anchor bottom) when a supplied photo
   exists, so the photo bleeds off the edge and fades into the page; InfoBar band (anchor bottom) as a footer band for deadline,
   contact or URL; IconRow for a list of 3-6 short items (domains, tracks, benefits) with an icon each; BigWord for a short
   verbatim word or number (a year, '30') as the graphic; Divider variant 'ornament' between info groups. A subtle abstract
   generated ground (soft light rays, folds, gradient texture; tone light, no objects) is welcome on event posters with little
   imagery. PHOTOS: PhotoFade/PhotoStrip may set treatment "duotone" (colour ramp in brand colours) to unify mixed or off-brand
   event photos; leave it off for clean photos. COMPOSITION: set top-level "align": "left" for an editorial, asymmetric poster (left-aligned title/body stacks, a
   left-aligned logo, Columns with the title block left and details right, optionally a left `stripe` decoration); keep
   "center" for formal, symmetric posters. Choose per event: formal ceremonies centre, recruitment/workshops/talks often left.
   Only use text, photos and QR codes that the brief or inventory actually provides. Keep related lines together and
   in the brief's order (e.g. the batch name and its years); a footer band holds only the deadline / contact / URL lines.
10. Brand logos need a flat light field. On a dark background give LogoBlock `plate: true` (a plate that hugs the logo). Do NOT wrap the
    logo in a full-width Panel: that creates a large empty box. Heroes grow into spare space automatically; do not pad with filler.
11. Keep text readable: light text ("on_primary"/"surface") only on dark fills such as "primary"; on light fills ("surface",
   "accent", gradients to white) use "primary" or "text". "muted" is only for secondary text on light grounds. Never put white
   text on "accent" fills.
12. If you receive PROBLEMS, make the SMALLEST change that resolves them: keep every other component, its content, order,
    emphasis and styling exactly as in YOUR PREVIOUS PLAN, and return the full corrected plan. Do not redesign.

OUTPUT: one JSON object {{"format": str, "background": {{"kind": "solid|gradient|generated", "color": str, "gradient": [{{"offset": 0..1, "color": str}}], "angle": number,
"art": {{"description": str, "style": str, "lighting": str, "medium": str, "tone": "light|dark", "palette": [str]}} (only for generated)}},
"generated_assets": [{{"id": str, "kind": "hero_cutout", "subject": str, "aspect": "1:1|4:5|3:4|16:9|9:16"}}],
"align": "center|left", "decorations": [{{"kind": "glow|dot_grid|rings|corner_shapes|diagonal_band|frame|stripe|watermark_text", "region": "full|top|bottom|left|right|center",
"color": str, "opacity": 0.02-0.6, "text": str (watermark_text only), "density": 0.4-3}}],
"components": [{{"id": str, "component": str, "variant": str, "content": {{...}}, "emphasis": str, "anchor": str, "children": [...]}}], "rationale": str}}.
Component ids must be unique, short and snake_case."""


def _brand_block(brand: BrandProfile) -> dict[str, Any]:
    return {
        "name": brand.name, "key": brand.key,
        "palette": brand.palette,
        "fonts": {k: f"{v.family} {v.style}" for k, v in brand.fonts.items()},
        "logos": {k: {"kind": v.kind, "language": v.language, "meant_for_field": v.field, "aspect": round(v.aspect, 2)}
                  for k, v in brand.logos.items()},
        "logo_rules": brand.logo_rules.model_dump(exclude={"forbidden"}),
        "background_rules": brand.background_rules,
        "forbidden_grounds": brand.forbidden_grounds, "allowed_decorations": brand.allowed_decorations,
        "footer_defaults": {k: v for k, v in brand.footer.model_dump().items() if v and k != "auto_include"},
    }


def _catalog_block() -> dict[str, Any]:
    return {n: {"desc": c["desc"], "variants": c["variants"], "content": c["content"], "required": c["required"]}
            for n, c in CATALOG.items()}


def _inventory_block(inventory: dict[str, InventoryItem]) -> list[dict[str, Any]]:
    return [{"id": i.id, "kind": i.kind.value, "label": i.label, "size": f"{i.width}x{i.height}",
             **({"warnings": i.warnings} if i.warnings else {})} for i in inventory.values()]


def build_user_prompt(brief: Brief, brand: BrandProfile, inventory: dict[str, InventoryItem], examples: list[dict] | None = None,
                      previous: dict | None = None, problems: list[str] | None = None) -> str:
    parts = [
        "FORMATS: " + json.dumps({k: f"{int(f.width_pt)}x{int(f.height_pt)} {f.kind}" for k, f in FORMATS.items()}),
        "BRAND:\n" + json.dumps(_brand_block(brand), separators=(",", ":")),
        "CATALOG:\n" + json.dumps(_catalog_block(), separators=(",", ":")),
        "ICONS (names for IconRow): " + ", ".join(icon_names()),
        "ASSET INVENTORY:\n" + json.dumps(_inventory_block(inventory), separators=(",", ":")),
        "BRIEF:\n" + json.dumps(brief.model_dump(exclude_none=True, exclude_defaults=True), separators=(",", ":"), ensure_ascii=False),
    ]
    for i, ex in enumerate(examples or []):
        parts.append(f"EXAMPLE PLAN {i + 1} (style reference from a real poster — do NOT copy its text):\n"
                     + _j(ex))
    if previous is not None and problems:
        parts.append("YOUR PREVIOUS PLAN:\n" + json.dumps(previous, separators=(",", ":"), ensure_ascii=False))
        parts.append("PROBLEMS (fix all):\n" + "\n".join(f"- {p}" for p in problems))
    parts.append("Return the plan JSON now.")
    return "\n\n".join(parts)
