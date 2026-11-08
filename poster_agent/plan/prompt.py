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
   quotation marks. Write art as a poster ART DIRECTOR, not a description: subject (one focal subject and where it sits), composition (camera angle,
   foreground / middle / background layers, where the quiet space is), devices (curved horizon, diagonal light beam, layered shapes: what makes it
   designable), color_roles (dominant, accent, light direction). Generic ("deep navy technology atmosphere") is wrong; specific ("a robot arm low in
   the right third, shot from below, with a diagonal blue light beam crossing a dark upper half") is right. The art must depict the event's own subject: a concrete scene or objects named by the brief's title, subtitle,
   tagline or style notes (use those nouns in art.description), not a generic abstract atmosphere. If you choose generated art set art.tone ("dark" art needs light text colours via "on_primary"/"surface").
9. Clean, professional institutional look; never decorate to fill space (no glows, dots, rings or random shapes). `decorations` is
   optional (border / frame / stripe only; a default border is applied). Fill space with scale and structure, following DESIGN SKILLS and
   the assigned COMPOSITION. Component hints: BigTitle (short title scaled to the width, optional pill), TabBanner (anchor top), Columns
   (children set content.column left|right), PhotoFade (anchor bottom; supplied photo; treatment "duotone" allowed), InfoBar band
   (anchor bottom: deadline / contact / URL lines only), IconRow (3-6 short items), BigWord (a short verbatim word or number),
   HeaderRow (logo + date/time/venue lines; pinned to the top edge); see CATALOG for the rest. A guest from the brief: ChiefGuestCard (label, name, designation lines; a supplied guest photo goes in it as a face in a circle), not plain lines. Top-level fields: template, align center|left, flow_valign,
   type_style. SKELETONS (template house | split | giant_word | greeting | photo_led) are single-column structures from the
   institution's posters: follow the COMPOSITION recipe; a BottomCutout / PhotoFade anchor at the bottom fills leftover height.
   Templates give named zones (stage: head, art, title, details; two_col: head, title, left, right; top_art: head, art,
   main; classic: no zones). With a non-classic template put every flow component in a zone, in reading order within it.
   Use the brand accent deliberately; text colours adapt to the real background automatically. Keep related lines together, in the
   brief's order. Only use text, photos and QR codes that the brief or inventory provides.
10. Header logo: use the WIDE full college logo (BRAND.logos 'rectangle') as a LogoBlock with width_pct about 46, left-aligned or centred, as on the institution's own posters; never the small square logo for the header, and no date / time / venue beside it. Logos: check BRAND.logos[...].solid_field. A logo with solid_field true (a navy square, for example) brings its own field: place it
    directly on any ground and NEVER add `plate`. Only a logo with solid_field false needs `plate: true` on a dark or busy ground. Do NOT
    wrap the logo in a full-width Panel: that creates a large empty box. Heroes grow into spare space automatically; no filler.
11. Keep text readable: light text ("on_primary"/"surface") only on dark fills such as "primary"; on light fills ("surface",
   "accent", gradients to white) use "primary" or "text". "muted" is only for secondary text on light grounds. Never put white
   text on "accent" fills.
12. DESIGN DEVICES (pick the ones that serve the idea, name them in concept.devices; never decorate for its own sake): two_tone_ground = background.panel, a solid
    brand-colour block from just above the flow component `starts_at` to the bottom edge (light top, dark information panel; text on it light); pull_quote = a
    PullQuote component for a quote or tribute line; keyline_frame = a "frame" or "border" decoration; big_type = BigTitle / BigWord / display TitleStack; hero_image = a
    photo or cutout; art_ground = generated art. The focal point is the one hero element; everything else supports it.
13. If you receive PROBLEMS, make the SMALLEST change that resolves them: keep every other component, its content, order,
    emphasis and styling exactly as in YOUR PREVIOUS PLAN, and return the full corrected plan. Do not redesign.

OUTPUT: one JSON object, keys in this order. FIRST "concept": {{"idea": str (the one idea, one sentence), "focal_point": str (id of the hero component), "reading_order": [component ids, eye path, focal point first], "devices": [from two_tone_ground, pull_quote, keyline_frame, big_type, hero_image, art_ground], "contrast": str (the main contrast: scale, colour, weight or space)}}; decide it before the layout and make the plan do exactly what it says. Then {{"format": str, "background": {{"kind": "solid|gradient|generated", "panel": {{"color": str, "starts_at": component id}} (optional two-tone ground), "color": str, "gradient": [{{"offset": 0..1, "color": str}}], "angle": number,
"art": {{"description": str, "subject": str, "composition": str, "devices": str, "color_roles": str, "style": str, "lighting": str, "medium": str, "tone": "light|dark", "palette": [str]}} (only for generated)}},
"generated_assets": [{{"id": str, "kind": "hero_cutout", "subject": str, "aspect": "1:1|4:5|3:4|16:9|9:16"}}],
"template": "house|split|giant_word|greeting|photo_led|classic|stage|two_col|top_art", "align": "center|left", "flow_valign": "top|balanced|center|bottom", "type_style": "classic|bold|elegant|editorial|condensed", "decorations": [{{"kind": "glow|dot_grid|rings|corner_shapes|diagonal_band|frame|stripe|watermark_text", "region": "full|top|bottom|left|right|center",
"color": str, "opacity": 0.02-0.6, "text": str (watermark_text only), "density": 0.4-3}}],
"components": [{{"id": str, "component": str, "variant": str, "content": {{...}}, "emphasis": str, "anchor": str, "zone": str (zone name of the template; omit for classic), "children": [...]}}], "rationale": str (the design concept: idea, visual metaphor, archetype, ground and palette roles, type style; never printed)}}.
Component ids must be unique, short and snake_case."""

# Composition recipes. Every poster is assigned one (deterministically from its title, or forced with brief.layout_style) so that
# different events get structurally different layouts instead of one logo-title-details stack. A recipe is guidance for the planner:
# it may deviate only when the content cannot support it.
RECIPES: dict[str, str] = {
    "header_row": 'align "left", type_style "bold" or "editorial". Open with HeaderRow (logo on the left, the date/time/venue lines '
                  'on the right), then a left-aligned BigTitle or TitleStack, then left-aligned body lines; contact/URL in a bottom '
                  'InfoBar band.',
    "title_low": 'flow_valign "bottom", align "left", type_style "condensed". Leave the upper half to the art (generated background, '
                 'PhotoFade or HeroImage); a small LogoBlock (align left) at the top; the big title and details in the lower third.',
    "big_word": 'align "center", type_style "elegant" or "bold". If the brief has a short number or word (a year, 30, JOIN), make it a BigWord '
                'hero with the logo small in a corner (LogoBlock align left) and the details in an InfoBar band; otherwise BigTitle.',
    "tab_and_rows": 'align "center", type_style "editorial". TabBanner (anchor top) with a strapline from the brief, LogoBlock align left, '
                    'BigTitle, then an IconRow or Columns for the list content, and an InfoBar band at the bottom.',
    "stage": 'template "stage": zone head = one HeaderRow (logo + date/time/venue; never a Panel); zone art = a HeroImage / ImageBlock / PhotoFade (a '
             'generated hero_cutout if no photo is supplied); zone title = BigTitle or TitleStack, sitting over the art\'s lower edge; zone details = the list '
             '(IconRow / TextBlocks); contact in a bottom InfoBar band. align "left", type_style "bold".',
    "columns_page": 'template "two_col": zone head = one HeaderRow (never a Panel); zone title = a wide BigTitle; zone left = date/time/venue and '
                    'body lines; zone right = a HeroImage / ImageBlock or an IconRow. align "left"; contact in a bottom InfoBar band.',
    "art_half": 'template "top_art": zone head = one HeaderRow (never a Panel); zone art = a tall HeroImage / ImageBlock (a generated hero_cutout if no '
                'photo is supplied); zone main = title, details and the CTA stacked together in the lower half. align "left" or "center".',
    "house": 'template "house", the institution\'s standard structure, top to bottom: (1) header strip: HeaderRow (logo + date/time/venue '
             'lines), or LogoStrip when partner logos exist, or LogoBlock; (2) the headline as wide as the poster: BigTitle for up to about '
             '5 words, else TitleStack (title_style for contrast); (3) a kicker in spaced capitals (TitleStack eyebrow / tagline); (4) a '
             'FactLine with date | time | venue unless the header already carries them; (5) ONE content block that fits the brief: '
             'SpeakerGrid / ChiefGuestCard for people, Columns for a day-wise outline, IconRow for 3-6 short items, FactStrip for fee / mode '
             '/ duration / eligibility, TextBlock for a short paragraph, QRBlock for a link; (6) the bottom anchor: a BottomCutout of a '
             'generated object (declare it in generated_assets) or a PhotoFade of a supplied photo, and/or an InfoBar band for contact / '
             'CTA lines. Light ground (surface) by default. The anchor takes all height the content leaves, so never add filler.',
    "split": 'template "split": a text column on the left (header, headline, kicker, FactLine, content) and a BottomCutout with beside: '
             'true, align right, standing on the bottom edge next to it (a generated object, or a supplied cutout of the speaker). align "left". '
             'InfoBar band optional.',
    "giant_word": 'template "giant_word": one giant verbatim word or number from the brief (a year, an edition, MAY DAY) as a BigWord '
                  'hero, the headline and kicker beside or under it, a FactLine, the logo at the top; InfoBar band optional.',
    "greeting": 'template "greeting" for festival / tribute posters: the logo at the top, one centred image (HeroImage of a generated '
                'object, or a supplied photo), the greeting as a TitleStack, almost nothing else.',
    "photo_led": 'template "photo_led" (only when a photo is supplied): the text at the top (header, headline, FactLine, content), the '
                 'photo as a PhotoFade with anchor bottom taking the lower part of the poster.',
    "classic": 'align "center", type_style "classic". Centred LogoBlock, TitleStack, details, optional band: use only when the content '
               'is formal (ceremony, convocation) or very short.',
}


# Archetypes that suit each genre (see skills/genre-playbooks); the brief's title picks one of them deterministically.
GENRE_RECIPES: dict[str, list[str]] = {  # the skeletons the institution's own posters use for each kind of event; the title picks one
    "workshop": ["house", "split", "giant_word"], "recruitment": ["house", "split", "giant_word"],
    "admission": ["house", "photo_led", "split"], "conference": ["house", "split", "giant_word"],
    "talk": ["split", "house", "photo_led"], "cultural": ["house", "giant_word", "split"], "sports": ["house", "photo_led", "giant_word"],
    "commemoration": ["greeting", "giant_word"], "placement": ["house", "split", "giant_word"], "launch": ["house", "giant_word", "split"],
}


def wants_icon_row(brief) -> bool:
    """The icon row is optional: about half the posters (chosen by the title, or forced with brief.icons) carry one."""
    import hashlib
    forced = getattr(brief, "icons", None)
    return forced if isinstance(forced, bool) else int(hashlib.sha1(("icons|" + brief.event_title).encode()).hexdigest(), 16) % 2 == 0


def icon_row_hint(brief) -> str:
    """One sentence for the planner prompt: use an IconRow or not."""
    return "Use an IconRow for the list lines." if wants_icon_row(brief) else "No IconRow: use Columns or a TextBlock."


def composition_recipe(brief) -> tuple[str, str]:
    """(name, guidance) for this brief: forced by brief.layout_style, else chosen deterministically from the event title."""
    import hashlib
    from poster_agent.skills import genre_of
    names = GENRE_RECIPES.get(genre_of(brief)) or ["house"]
    if not getattr(brief, "photo_assets", None):  # a photo-led poster needs a supplied photo
        names = [n for n in names if n != "photo_led"] or ["house"]
    forced = getattr(brief, "layout_style", None)
    name = forced if forced in RECIPES else names[int(hashlib.sha1(brief.event_title.encode()).hexdigest(), 16) % len(names)]
    return name, RECIPES[name]


def _look_text(brief) -> str:
    from .looks import look_for
    lk = look_for(brief)
    return f"{lk.name} = {lk.about}"


def skills_text(brief) -> str:
    """The design guidance relevant to this brief (genre playbook + digests), small enough for the planner prompt."""
    from poster_agent.skills import select
    return select(brief, budget_chars=2400)[0] or "(no skills installed)"


def _brand_block(brand: BrandProfile) -> dict[str, Any]:
    return {
        "name": brand.name, "key": brand.key,
        "palette": brand.palette,
        "fonts": {k: f"{v.family} {v.style}" for k, v in brand.fonts.items()},
        "logos": {k: {"kind": v.kind, "language": v.language, "meant_for_field": v.field, "aspect": round(v.aspect, 2),
                       "solid_field": v.solid_field}
                  for k, v in brand.logos.items()},
        "logo_rules": brand.logo_rules.model_dump(exclude={"forbidden"}),
        "background_rules": brand.background_rules,
        "forbidden_grounds": brand.forbidden_grounds, "allowed_decorations": brand.allowed_decorations,
        "footer_defaults": {k: v for k, v in brand.footer.model_dump().items() if v and k != "auto_include"},
    }


def _relevant(brief, inventory) -> set[str]:
    """Components worth describing to the planner for this brief: the always-useful core plus the ones its inventory / fields can feed."""
    kinds = {i.kind.value for i in (inventory or {}).values()}
    names = set(CATALOG) - {"PhotoFade", "PhotoStrip", "ImageBlock", "SpeakerGrid", "ChiefGuestCard", "PartnerLogoRow", "LogoStrip",
                            "LogoTitleRow"}
    if kinds & {"photo", "texture", "generated_background"} or getattr(brief, "photo_assets", None):
        names |= {"PhotoFade", "PhotoStrip", "ImageBlock"}
    if "headshot" in kinds or getattr(brief, "chief_guest", None) or getattr(brief, "speakers", None):
        names |= {"SpeakerGrid", "ChiefGuestCard"}
    if "partner_logo" in kinds or getattr(brief, "partner_assets", None):
        names |= {"PartnerLogoRow", "LogoStrip", "LogoTitleRow"}
    return names


def _catalog_block(brief=None, inventory=None) -> dict[str, Any]:
    keep = _relevant(brief, inventory) if brief is not None else set(CATALOG)
    return {n: {"desc": c["desc"], "variants": c["variants"], "content": c["content"], "required": c["required"]}
            for n, c in CATALOG.items() if n in keep}


def _inventory_block(inventory: dict[str, InventoryItem]) -> list[dict[str, Any]]:
    return [{"id": i.id, "kind": i.kind.value, "label": i.label, "size": f"{i.width}x{i.height}",
             **({"warnings": i.warnings} if i.warnings else {})} for i in inventory.values()]


def build_user_prompt(brief: Brief, brand: BrandProfile, inventory: dict[str, InventoryItem], examples: list[dict] | None = None,
                      previous: dict | None = None, problems: list[str] | None = None) -> str:
    parts = [
        "FORMATS: " + json.dumps({k: f"{int(f.width_pt)}x{int(f.height_pt)} {f.kind}" for k, f in FORMATS.items()}),
        "BRAND:\n" + json.dumps(_brand_block(brand), separators=(",", ":")),
        "CATALOG:\n" + json.dumps(_catalog_block(brief, inventory), separators=(",", ":")),
        *(["ICONS (names for IconRow): " + ", ".join(icon_names())] if len(brief.extra_lines) + len(brief.theme_lines) >= 3 else []),
        "ASSET INVENTORY:\n" + json.dumps(_inventory_block(inventory), separators=(",", ":")),
        "COMPOSITION FOR THIS POSTER (recipe '%s'; follow it unless the content makes it impossible): %s" % composition_recipe(brief),
        "LOOK of this poster (only its ground and type style are applied to your plan; YOU choose the header, footer, text colours (light on "
        "a dark ground, dark on a light one) and headline treatment): "
        "%s. Generated art is a full-bleed ground; a cutout may stand on it. %s" % (_look_text(brief), icon_row_hint(brief)),
        "DEFAULT FORMAT: use \"story\" (84 %% of the catalogue) when the brief gives no format.",
        "DESIGN SKILLS (guidance for the concept and the plan; they add no copy, every printed word is from the brief):" + chr(10) + skills_text(brief),
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
