"""Component base + layout context. A component answers two questions: how tall am I at this width
(`measure`), and which exact elements do I emit inside the box I was given (`place`)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from poster_agent.spec import (AssetKind, AssetRef, Box, CanvasSpec, ComponentSpec, ElementKind, Issue,
                               IssueCategory, PlacedElement)
from poster_agent.spec.brand import BrandProfile, FontRole

from .fonts import TextMeasurer

BALANCED_ROLES = {"display", "title", "heading", "subtitle"}  # roles whose line breaks are balanced (headlines, not body copy)
WRAP_SAFETY = 0.97  # measurer vs VectorCraft differ by up to ~1.4%; wrap with margin to stay clear of it

SECONDARY_UP_MAX = 1.12  # supporting text may grow at most this much when the solver fills spare space
ROLE_MAX_FRACTION = {"heading": 0.042, "subtitle": 0.036, "body": 0.030, "caption": 0.024, "label": 0.022, "micro": 0.019}

# role -> (font role, size as fraction of canvas width, leading ratio, tracking 1/1000 em)
TYPE_SCALE: dict[str, tuple[str, float, float, float]] = {
    "display":  ("display_heavy", 0.100, 1.05, 0),
    "title":    ("display", 0.050, 1.15, 0),
    "heading":  ("display", 0.036, 1.2, 0),
    "subtitle": ("body", 0.030, 1.25, 0),
    "body":     ("body", 0.026, 1.3, 0),
    "caption":  ("body", 0.021, 1.25, 0),
    "label":    ("display", 0.020, 1.2, 150),
    "micro":    ("body", 0.0165, 1.2, 0),
}


# Typography presets: per type role, override the font role, case, tracking and a size factor. A component that sets its own
# font/tracking/case still wins. Roles a brand does not define fall back to its display/body fonts.
TYPE_STYLES: dict[str, dict[str, dict]] = {
    "classic": {},
    "bold": {"display": {"font": "display_heavy", "upper": True}, "title": {"font": "display_heavy", "upper": True},
             "heading": {"font": "display", "upper": True}, "label": {"font": "display", "tracking": 200, "upper": True}},
    "elegant": {"display": {"font": "light", "upper": True, "tracking": 120}, "title": {"font": "light", "upper": True, "tracking": 160},
                "heading": {"font": "light", "upper": True, "tracking": 120}, "subtitle": {"font": "light"},
                "label": {"font": "display", "tracking": 300, "upper": True}},
    "editorial": {"label": {"font": "display", "tracking": 300, "upper": True}, "subtitle": {"font": "light", "factor": 1.1},
                  "body": {"font": "light", "factor": 1.05}, "title": {"font": "display"}},
    "condensed": {"display": {"font": "condensed", "upper": True, "factor": 1.25}, "title": {"font": "condensed", "upper": True, "factor": 1.2},
                  "heading": {"font": "condensed", "upper": True, "factor": 1.15}, "subtitle": {"font": "condensed_regular", "factor": 1.1},
                  "label": {"font": "condensed", "tracking": 200, "upper": True}},
}


@dataclass
class Ctx:
    brand: BrandProfile
    canvas: CanvasSpec
    measurer: TextMeasurer
    assets: dict[str, AssetRef]
    scale: float = 1.0  # global shrink applied by the solver when content does not fit
    type_style: str = "classic"  # typography preset (TYPE_STYLES)
    align: str = "center"  # the poster's composition: 'center' (symmetric) or 'left' (editorial, left-aligned stacks)
    ground: str | None = None  # the poster's real background colour (darkest gradient stop); derived colours are made readable on it
    extra: dict[str, float] = field(default_factory=dict)  # component id -> spare height (pt) an elastic component may grow by
    issues: list[Issue] = field(default_factory=list)

    # ---- palette / fonts -------------------------------------------------
    def color(self, name: str) -> str:
        return resolve_color(self.brand, name, self.ground)

    def font(self, role: str) -> FontRole:
        f = self.brand.fonts
        return f.get(role) or f["display" if role.startswith("display") else "body"]

    def floor(self, role: str) -> float:
        h = self.canvas.height
        return max(14.0, {"display": 0.04, "title": 0.025, "heading": 0.02, "micro": 0.0}.get(role, 0.0135) * h)

    def size(self, role: str, factor: float = 1.0) -> float:
        _, frac, _, _ = TYPE_SCALE[role]
        # When the solver scales UP to fill space, only the title-class roles take the full growth; supporting text grows a little at most
        up = self.scale if (role in ("display", "title") or self.scale <= 1.0) else min(self.scale, SECONDARY_UP_MAX)
        size = frac * self.canvas.width * factor * up
        cap = ROLE_MAX_FRACTION.get(role)  # supporting roles never get bigger than this share of the poster width
        if cap is not None:
            size = min(size, cap * self.canvas.width * max(factor, 1.0))
        return max(self.floor(role), size)

    # ---- assets ---------------------------------------------------------
    def brand_logo_asset(self, name: str | None = None) -> str:
        name = name or self.brand.default_logo
        aid = f"brand_logo_{name}"
        if aid not in self.assets:
            self.assets[aid] = AssetRef(id=aid, kind=AssetKind.logo, path=self.brand.logo(name).path, provenance="brand")
        return aid

    def issue(self, category: IssueCategory, evidence: str, ids: list[str] | None = None, fix: str | None = None,
              severity: str = "error") -> None:
        self.issues.append(Issue(category=category, severity=severity, element_ids=ids or [], evidence=evidence,
                                 suggested_fix=fix, source="solver"))

    # ---- text -----------------------------------------------------------
    def text(self, id_: str, role: str, text: str, x: float, y: float, w: float, *, color: str = "text",
             align: str = "left", font_role: str | None = None, size: float | None = None, factor: float = 1.0,
             tracking: float | None = None, z: int = 10, source: str | None = None,
             max_lines: int | None = None, upper: bool = False) -> tuple[PlacedElement, float]:
        """One wrapped text element. Returns (element, height). Shrinks to `max_lines` (never below the floor)."""
        frole, _, lead_ratio, trk = TYPE_SCALE[role]
        st = TYPE_STYLES.get(self.type_style, {}).get(role, {})
        font = self.ctx_font(font_role or st.get("font", frole))
        trk = (st.get("tracking", trk)) if tracking is None else tracking
        upper = upper or bool(st.get("upper"))
        factor = factor * st.get("factor", 1.0) if size is None else factor
        text = text.upper() if upper else text
        size = size or self.size(role, factor)
        floor = self.floor(role)
        wrap = self.measurer.wrap_balanced if role in BALANCED_ROLES else self.measurer.wrap  # headlines break into even lines
        lines = wrap(text, font.family, font.style, size, w * WRAP_SAFETY, trk)
        while max_lines and len(lines) > max_lines and size > floor:
            size = max(floor, size * 0.94)
            lines = wrap(text, font.family, font.style, size, w * WRAP_SAFETY, trk)
        widest = self.measurer.widest(lines, font.family, font.style, size, trk)
        while widest > w * WRAP_SAFETY and size > floor:  # one word wider than its box (a long label in a narrow cell): shrink, don't fail
            size = max(floor, size * 0.94)
            lines = wrap(text, font.family, font.style, size, w * WRAP_SAFETY, trk)
            widest = self.measurer.widest(lines, font.family, font.style, size, trk)
        if widest > w:
            self.issue(IssueCategory.fit, f"{id_}: longest line {widest:.0f}pt exceeds width {w:.0f}pt at {size:.0f}pt",
                       [id_], "shorten the text or give the component more width")
        if max_lines and len(lines) > max_lines:
            self.issue(IssueCategory.fit, f"{id_}: {len(lines)} lines > max {max_lines} at the {floor:.0f}pt floor",
                       [id_], "shorten the text")
        leading = size * lead_ratio
        h = len(lines) * leading + 0.2 * size
        props: dict[str, Any] = {"leading": round(leading, 2)}
        if trk:
            props["tracking"] = trk
        el = PlacedElement(id=id_, kind=ElementKind.text, role=role, box=Box(x=x, y=y, w=w, h=h), z=z, text="\n".join(lines),
                           font=font.family, style=font.style, size=round(size, 2), color=self.color(color), align=align,
                           props=props, source_component=source)
        return el, h

    def ctx_font(self, role: str) -> FontRole:
        return self.font(role)

    def style_font(self, role: str, default: str) -> str:
        """The font role the typography preset gives a type role (for components that pick fonts themselves, like BigTitle)."""
        return TYPE_STYLES.get(self.type_style, {}).get(role, {}).get("font", default)


def resolve_color(brand: BrandProfile, name: str, ground: str | None = None) -> str:
    """Palette name (or hex) -> hex. Names: any brand palette key, plus derived 'muted' (readable on `ground`, default the brand
    surface) and 'on_primary'."""
    if name.startswith("#"):
        return name
    p = brand.palette
    if name == "muted":
        return p.get("muted") or _muted(p["text"], ground or p["surface"])
    if name == "on_primary":
        return p["surface"]
    return p.get(name) or p["text"]


def valid_color_names(brand: BrandProfile) -> set[str]:
    return set(brand.palette) | {"muted", "on_primary"}


def _muted(text: str, surface: str, floor: float = 5.0) -> str:
    """A softened text colour: mix text toward `surface` (the real ground) as far as it stays >= `floor`:1 against it. The floor sits
    above WCAG AA (4.5) so subtle tints and decorations behind the text cannot push it under the line."""
    from poster_agent.checks.pixels import contrast_ratio, hex_rgb, rel_luminance
    ls = rel_luminance(hex_rgb(surface))
    best = text
    for t in (0.1 * i for i in range(1, 6)):  # 0.1 .. 0.5
        c = _mix(text, surface, t)
        if contrast_ratio(rel_luminance(hex_rgb(c)), ls) >= floor:
            best = c
        else:
            break
    return best


def _mix(a: str, b: str, t: float) -> str:
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x * (1 - t) + y * t):02x}" for x, y in zip(ca, cb))


class Component(ABC):
    """Implement `build` only: measure() and place() derive from it, so they can never disagree."""
    name: ClassVar[str]
    variants: ClassVar[tuple[str, ...]] = ("default",)
    elastic: ClassVar[bool] = False  # True: may grow into spare vertical space (the solver grants it via ctx.extra)

    @abstractmethod
    def build(self, spec: ComponentSpec, ctx: Ctx, x: float, y: float, w: float) -> tuple[list[PlacedElement], float]:
        """Elements for this component laid out in a column at (x, y) of width w; returns (elements, height)."""

    def measure(self, spec: ComponentSpec, ctx: Ctx, width: float) -> float:
        return self.build(spec, ctx, 0.0, 0.0, width)[1]

    def place(self, spec: ComponentSpec, ctx: Ctx, box: Box) -> list[PlacedElement]:
        return self.build(spec, ctx, box.x, box.y, box.w)[0]


REGISTRY: dict[str, Component] = {}


def register(cls):
    REGISTRY[cls.name] = cls()
    return cls
