"""Deterministic validation of an LLM plan. Returns plain-English errors the planner can act on (they are sent back verbatim)."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import yaml

from poster_agent.ingest import Brief, InventoryItem
from poster_agent.layout.base import valid_color_names
from poster_agent.layout.catalog import ANCHORS, CATALOG, EMPHASIS, TYPE_ROLES
from poster_agent.spec import DECORATION_KINDS, NAMED_REGIONS, ComponentSpec, get_format, normalize
from poster_agent.spec.brand import ROOT, BrandProfile

from .schema import PlanOutput

SEPARATORS = re.compile(r"\s*[|•·\n]\s*")  # a block may stack several brief lines, one per line
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
LOGO_COMPONENTS = {"LogoBlock", "LogoTitleRow"}
# Generated imagery must describe objects/atmosphere only: no printed text, logos or people (guardrail 4).
FORBIDDEN_ART = re.compile(r"\b(text|typograph\w*|letter\w*|words?|writing|captions?|logos?|brands?|"
                           r"person|people|man|men|woman|women|face|faces|portrait\w*|human\w*|students?|crowd\w*)\b", re.I)
QUOTES = re.compile("[\"'“”]")


def walk(components: list[ComponentSpec]) -> Iterator[ComponentSpec]:
    for c in components:
        yield c
        yield from walk(c.children)


@lru_cache(maxsize=1)
def _structural() -> tuple[set[str], list[re.Pattern]]:
    cfg = yaml.safe_load((ROOT / "configs/structural_labels.yaml").read_text(encoding="utf-8"))
    return {normalize(x) for x in cfg["labels"]}, [re.compile(p) for p in cfg.get("patterns", [])]


def is_structural_label(text: str) -> bool:
    labels, patterns = _structural()
    n = normalize(text)
    return n in labels or any(p.match(n) for p in patterns)


def _get(content: dict, key: str) -> list[Any]:
    """Values at a catalog key: 'a' -> content['a']; 'a[].b' -> [x['b'] for x in content['a']]."""
    if "[]." in key:
        head, tail = key.split("[].")
        return [x.get(tail) for x in content.get(head) or [] if isinstance(x, dict)]
    v = content.get(key)
    return v if isinstance(v, list) else [v]


def component_texts(c: ComponentSpec) -> list[str]:
    out: list[str] = []
    for key in CATALOG.get(c.component, {}).get("text_keys", []):
        out += [v for v in _get(c.content, key) if isinstance(v, str) and v.strip()]
    return out


def component_assets(c: ComponentSpec) -> list[str]:
    out: list[str] = []
    for key in CATALOG.get(c.component, {}).get("asset_keys", []):
        out += [v for v in _get(c.content, key) if isinstance(v, str) and v]
    return out


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", normalize(text))


def validate_plan(plan: PlanOutput, brief: Brief, brand: BrandProfile, inventory: dict[str, InventoryItem]) -> list[str]:
    errs: list[str] = []
    try:
        get_format(plan.format)
    except ValueError as e:
        errs.append(str(e))

    colors, fonts = valid_color_names(brand), set(brand.fonts)
    banned = set(brand.forbidden_grounds) | {brand.palette[n].lower() for n in brand.forbidden_grounds if n in brand.palette}
    bgp = plan.background
    for v in [bgp.color] + [s.color for s in bgp.gradient]:
        if v.lower() in banned:
            errs.append(f"background colour {v!r} is not allowed as a ground for {brand.short_name} (brand rule); "
                        "use 'primary' or 'surface'")
    corpus = [normalize(t) for t in brief.corpus()]
    for f in (brand.footer.website, brand.footer.phone, brand.footer.address, brand.footer.email):
        if f:
            corpus.append(normalize(f))

    if len(plan.decorations) > 4:
        errs.append(f"at most 4 decorations (got {len(plan.decorations)})")
    for i, d in enumerate(plan.decorations):
        w = f"decoration {i} ({d.kind})"
        if d.kind not in DECORATION_KINDS or d.kind not in brand.allowed_decorations:
            errs.append(f"{w}: kind must be one of {sorted(set(DECORATION_KINDS) & set(brand.allowed_decorations))}")
        if d.region not in NAMED_REGIONS:
            errs.append(f"{w}: region {d.region!r} must be one of {list(NAMED_REGIONS)}")
        if d.color not in colors and not HEX.match(d.color):
            errs.append(f"{w}: colour {d.color!r} is not a palette name {sorted(colors)} or #rrggbb")
        if not 0.02 <= d.opacity <= 0.6:
            errs.append(f"{w}: opacity {d.opacity} must be between 0.02 and 0.6 (decoration stays subtle)")
        if not 0.4 <= d.density <= 3.0:
            errs.append(f"{w}: density {d.density} must be between 0.4 and 3.0")
        if d.kind == "watermark_text":
            if not d.text or not any(normalize(d.text) in src for src in corpus):
                errs.append(f"{w}: text {d.text!r} must be given and verbatim from the brief (e.g. the year)")
        elif d.text:
            errs.append(f"{w}: only watermark_text takes text")
    seen_ids: set[str] = set()
    logo_count = 0
    plan_tokens: list[str] = []
    used_assets: set[str] = set()
    all_components = list(walk(plan.components))

    gen_ids = {g.id for g in plan.generated_assets}
    for g in plan.generated_assets:
        if g.id in inventory:
            errs.append(f"generated asset id {g.id!r} collides with a supplied asset id")
        for bad in sorted({m.group(0).lower() for m in FORBIDDEN_ART.finditer(g.subject)}):
            errs.append(f"generated asset {g.id!r}: subject mentions {bad!r}; describe an object only "
                        "(no text, logos or people: those are never generated)")
        if QUOTES.search(g.subject):
            errs.append(f"generated asset {g.id!r}: subject must not contain quotation marks")
    bg = plan.background
    if bg.kind == "generated":
        if bg.art is None:
            errs.append("background kind 'generated' needs background.art")
        else:
            texts = [bg.art.description, bg.art.style, bg.art.lighting, bg.art.medium]
            for bad in sorted({m.group(0).lower() for t in texts for m in FORBIDDEN_ART.finditer(t)}):
                errs.append(f"background.art mentions {bad!r}; describe imagery and atmosphere only "
                            "(no text, logos or people)")
            if any(QUOTES.search(t) for t in texts):
                errs.append("background.art must not contain quotation marks")
            for pc in bg.art.palette:
                if pc not in colors and not HEX.match(pc):
                    errs.append(f"background.art.palette colour {pc!r} is not a palette name or #rrggbb")

    for c in all_components:
        where = f"component {c.id!r} ({c.component})"
        if c.id in seen_ids:
            errs.append(f"{where}: duplicate id")
        seen_ids.add(c.id)
        spec = CATALOG.get(c.component)
        if not spec:
            errs.append(f"{where}: unknown component; use one of {sorted(CATALOG)}")
            continue
        if c.variant not in spec["variants"]:
            errs.append(f"{where}: variant {c.variant!r} not in {spec['variants']}")
        if c.emphasis not in EMPHASIS:
            errs.append(f"{where}: emphasis {c.emphasis!r} not in {EMPHASIS}")
        if c.anchor not in ANCHORS:
            errs.append(f"{where}: anchor {c.anchor!r} not in {ANCHORS}")
        unknown = set(c.content) - set(spec["content"]) - {"column"}  # `column` is the placement key of a Columns child
        if unknown:
            errs.append(f"{where}: unknown content keys {sorted(unknown)}; allowed: {sorted(spec['content'])}")
        for req in spec["required"]:
            if not c.content.get(req):
                errs.append(f"{where}: missing required content key {req!r}")
        if c.component == "InfoBar" and c.anchor != "flow":
            body = {t.strip().lower() for t in [brief.event_title, brief.tagline, brief.subtitle, *brief.extra_lines] if t}
            stray = [i for i in c.content.get("items", []) if isinstance(i, str) and i.strip().lower() in body]
            if stray:
                errs.append(f"{where}: an edge band holds date/time/venue/contact lines only; {stray} are headline/body lines from the "
                            "brief: put them in a TextBlock, TitleStack or IconRow next to the related text")
        if c.content.get("treatment") is not None:
            if c.component not in ("PhotoFade", "PhotoStrip"):
                errs.append(f"{where}: treatment is only for PhotoFade / PhotoStrip (never logos or people portraits)")
            elif c.content["treatment"] not in brand.photo_treatments:
                errs.append(f"{where}: treatment {c.content['treatment']!r} not in the brand's allowed {brand.photo_treatments}")
        if c.component == "IconRow":
            from poster_agent.layout.structure import icon_names
            known = set(icon_names())
            for it in c.content.get("items") or []:
                if not isinstance(it, dict) or it.get("icon") not in known:
                    errs.append(f"{where}: icon {(it or {}).get('icon') if isinstance(it, dict) else it!r} is not in ICONS")
        if c.children and c.component not in ("Panel", "Columns"):
            errs.append(f"{where}: only Panel and Columns may have children")
        if c.component == "LogoBlock" and bgp.kind == "generated" and not c.content.get("plate"):
            errs.append(f"{where}: the background is generated art (textured, not a flat field): brand logos need a flat light "
                        "field, so set plate: true on the LogoBlock")
        if c.component in LOGO_COMPONENTS:
            logo_count += 1
            name = c.content.get("logo")
            if name and name not in brand.logos:
                errs.append(f"{where}: logo {name!r} not in brand logos {sorted(brand.logos)}")
        for key in spec["color_keys"]:
            v = c.content.get(key)
            if key in ("fill", "fill_to", "plate") and isinstance(v, str) and v.lower() in banned:
                errs.append(f"{where}: {key}={v!r} is not allowed as a background/fill for {brand.short_name} (brand rule: "
                            "the accent colour is not a ground); use 'primary' or 'surface'")
            if isinstance(v, str) and v not in colors and not HEX.match(v):
                errs.append(f"{where}: {key}={v!r} is not a palette colour name {sorted(colors)} or #rrggbb")
        for key in spec["font_keys"]:
            v = c.content.get(key)
            if isinstance(v, str) and v not in fonts:
                errs.append(f"{where}: {key}={v!r} is not a brand font role {sorted(fonts)}")
        if isinstance(c.content.get("role"), str) and c.content["role"] not in TYPE_ROLES:
            errs.append(f"{where}: role {c.content['role']!r} not in {TYPE_ROLES}")
        for aid in component_assets(c):
            used_assets.add(aid)
            if aid not in inventory and aid not in gen_ids:
                errs.append(f"{where}: asset {aid!r} is neither in the inventory {sorted(inventory)} nor declared in "
                            f"generated_assets {sorted(gen_ids)}")
        for text in component_texts(c):  # guardrail 3: verbatim
            for seg in (s for s in SEPARATORS.split(text) if s.strip()):
                n = normalize(seg)
                if not any(n in src for src in corpus) and not is_structural_label(seg):
                    errs.append(f"{where}: text {seg!r} is not in the brief (poster text must be verbatim from the brief)")
            plan_tokens += tokens(text)

    if logo_count != 1:
        errs.append(f"the poster needs exactly one institution logo component (LogoBlock or LogoTitleRow); found {logo_count}")

    have = set(plan_tokens)
    for path, text in brief.required_fields():  # coverage: nothing the requester gave may be dropped
        missing = [t for t in tokens(text) if t not in have]
        if missing:
            errs.append(f"brief field {path} ({text!r}) is not fully on the poster; missing words: {missing[:6]}")

    for aid in sorted(brief.asset_ids() - used_assets):
        errs.append(f"supplied asset {aid!r} is never used")
    for aid in sorted(gen_ids - used_assets):
        errs.append(f"generated asset {aid!r} is declared but no component uses it")
    return errs
