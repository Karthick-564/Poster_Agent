"""PosterSpec: the single source of truth every stage reads/writes (guardrail 2).

Plan stage fills `components`/`background`/asset needs; Layout fills `placed`; Compile draws `placed`;
Check/Critique append `issues`. No stage edits another stage's section — conflicts become Issues.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .formats import MM_TO_PT, FormatSpec, get_format


class Box(BaseModel):
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def intersects(self, other: "Box") -> bool:
        return self.x < other.right and other.x < self.right and self.y < other.bottom and other.y < self.bottom

    def intersection_area(self, other: "Box") -> float:
        dx = min(self.right, other.right) - max(self.x, other.x)
        dy = min(self.bottom, other.bottom) - max(self.y, other.y)
        return max(dx, 0.0) * max(dy, 0.0)

    def contains(self, other: "Box", tol: float = 0.0) -> bool:
        return (other.x >= self.x - tol and other.y >= self.y - tol
                and other.right <= self.right + tol and other.bottom <= self.bottom + tol)

    def inset(self, d: float) -> "Box":
        return Box(x=self.x + d, y=self.y + d, w=max(self.w - 2 * d, 0.0), h=max(self.h - 2 * d, 0.0))

    def expand(self, d: float) -> "Box":
        return self.inset(-d)


class CanvasSpec(BaseModel):
    format: str
    width: float
    height: float
    export_scale: float
    kind: Literal["digital", "print"] = "digital"
    bleed_pt: float = 0.0
    safe_margin_pct: float = 4.5

    @classmethod
    def from_format(cls, name: str, safe_margin_pct: float = 4.5) -> "CanvasSpec":
        f: FormatSpec = get_format(name)
        return cls(format=f.name, width=f.width_pt, height=f.height_pt, export_scale=f.export_scale,
                   kind=f.kind, bleed_pt=f.bleed_mm * MM_TO_PT, safe_margin_pct=safe_margin_pct)

    @property
    def safe_box(self) -> Box:
        m = self.safe_margin_pct / 100.0 * min(self.width, self.height)
        return Box(x=m, y=m, w=self.width - 2 * m, h=self.height - 2 * m)

    @property
    def full_box(self) -> Box:
        return Box(x=0, y=0, w=self.width, h=self.height)


class AssetKind(StrEnum):
    logo = "logo"
    partner_logo = "partner_logo"
    photo = "photo"
    headshot = "headshot"
    texture = "texture"
    generated_background = "generated_background"
    generated_hero = "generated_hero"
    icon = "icon"


class AssetRef(BaseModel):
    id: str
    kind: AssetKind
    path: str  # relative to project root
    provenance: Literal["brand", "supplied", "archive", "generated"] = "supplied"
    treatment: dict[str, Any] = Field(default_factory=dict)  # e.g. {"crop": "circle", "tone": "grayscale"}
    seed: int | None = None  # generated assets only (reproducibility)
    prompt: str | None = None

    @model_validator(mode="after")
    def _generated_only_for_art(self) -> "AssetRef":
        # Guardrail 4: logos/photos/headshots are never diffusion output.
        if self.provenance == "generated" and self.kind in {AssetKind.logo, AssetKind.partner_logo,
                                                            AssetKind.photo, AssetKind.headshot}:
            raise ValueError(f"{self.kind} assets must not be generated")
        return self


class GradientStop(BaseModel):
    offset: float = Field(ge=0.0, le=1.0)
    color: str
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)


class ScrimSpec(BaseModel):
    """Soft gradient overlay to protect text over art/photos."""
    edge: Literal["top", "bottom", "left", "right"] = "bottom"  # canvas edge the scrim is densest at
    color: str = "#000000"
    opacity: float = Field(default=0.6, ge=0.0, le=1.0)  # opacity at the edge, fading to 0
    extent: float = Field(default=0.5, gt=0.0, le=1.0)  # fraction of the canvas height/width it reaches in


class BackgroundSpec(BaseModel):
    kind: Literal["solid", "gradient", "texture", "generated", "photo"] = "solid"
    color: str = "#FFFFFF"
    gradient: list[GradientStop] = Field(default_factory=list)
    angle: float = 90.0
    asset_id: str | None = None
    scrims: list[ScrimSpec] = Field(default_factory=list)
    # Zones the layout needs kept calm when the background is generated (fed to the image prompt).
    keep_clear: list[Box] = Field(default_factory=list)


class ComponentSpec(BaseModel):
    """Planner output: *what* goes on the poster. Never coordinates."""
    id: str
    component: str  # registry name, e.g. "TitleStack", "SpeakerGrid"
    variant: str = "default"
    content: dict[str, Any] = Field(default_factory=dict)
    asset_ids: list[str] = Field(default_factory=list)
    emphasis: Literal["hero", "primary", "secondary", "tertiary"] = "primary"
    zone: str | None = None  # named region of the poster's layout template this component belongs to (flow components only)
    anchor: Literal["flow", "top", "bottom"] = "flow"  # top/bottom: full-bleed bands pinned to a canvas edge
    children: list["ComponentSpec"] = Field(default_factory=list)  # containers (Panel) lay these out inside


class ElementKind(StrEnum):
    text = "text"
    image = "image"
    shape = "shape"
    qr = "qr"


class PlacedElement(BaseModel):
    """Layout output: exact geometry. The compiler draws these and nothing else."""
    id: str
    kind: ElementKind
    role: str  # e.g. "title", "date", "logo", "speaker_name"
    box: Box
    z: int = 0
    source_component: str | None = None
    text: str | None = None
    font: str | None = None
    style: str | None = None  # font style, e.g. "Bold"
    size: float | None = None
    color: str | None = None
    align: Literal["left", "center", "right"] = "left"
    asset_id: str | None = None
    props: dict[str, Any] = Field(default_factory=dict)  # fill, radius, stroke, crop...
    locked: bool = False  # brand-locked (logos): checks may flag but nobody may move/resize

    @model_validator(mode="after")
    def _kind_requirements(self) -> "PlacedElement":
        if self.kind == ElementKind.text and not self.text:
            raise ValueError(f"text element {self.id!r} has no text")
        if self.kind == ElementKind.image and not self.asset_id:
            raise ValueError(f"image element {self.id!r} has no asset_id")
        return self


class MandatoryItem(BaseModel):
    """Something the finished poster MUST contain; verified on the rendered output (OCR/template match)."""
    id: str
    kind: Literal["text", "logo", "qr"] = "text"
    label: str
    text: str | None = None  # exact expected string for text items
    asset_id: str | None = None  # for logos


class Stage(StrEnum):
    ingest = "ingest"
    plan = "plan"
    layout = "layout"
    assets = "assets"
    compile = "compile"


class IssueCategory(StrEnum):
    mandatory_missing = "mandatory_missing"
    text_overlap = "text_overlap"
    margin_violation = "margin_violation"
    low_contrast = "low_contrast"
    logo_violation = "logo_violation"
    stray_text = "stray_text"
    output_spec = "output_spec"
    hierarchy = "hierarchy"
    density = "density"
    component_choice = "component_choice"
    spacing = "spacing"
    alignment = "alignment"
    fit = "fit"
    bg_clash = "bg_clash"
    hero_quality = "hero_quality"
    photo_treatment = "photo_treatment"
    render_artifact = "render_artifact"
    dead_space = "dead_space"


class Issue(BaseModel):
    category: IssueCategory
    severity: Literal["error", "warning", "info"] = "error"
    element_ids: list[str] = Field(default_factory=list)
    evidence: str  # what was measured/seen — never a claim without a measurement
    suggested_fix: str | None = None
    source: Literal["solver", "check", "critic"] = "check"
    round: int = 0


DECORATION_KINDS = ("glow", "dot_grid", "rings", "corner_shapes", "diagonal_band", "frame", "stripe", "watermark_text", "border")
NAMED_REGIONS = ("full", "top", "bottom", "left", "right", "center")


class Decoration(BaseModel):
    """Brand-native decoration drawn between the background and the content (vector shapes; never covers content).
    Expanded into placed elements by the layout stage, so it always sits behind text, logos and images."""
    id: str
    kind: Literal["glow", "dot_grid", "rings", "corner_shapes", "diagonal_band", "frame", "stripe", "watermark_text", "border"]
    region: Box | str = "full"  # a named region (full/top/bottom/left/right/center) or an explicit box in pt
    color: str = "accent"  # palette name or #hex
    opacity: float = Field(default=0.15, ge=0.02, le=1.0)
    text: str | None = None  # watermark_text only: must be verbatim from the brief
    density: float = Field(default=1.0, ge=0.4, le=3.0)  # dot spacing / ring count multiplier
    source: Literal["plan", "auto"] = "plan"  # 'auto' = added by the enrich repair after measuring dead space


class ElementOverride(BaseModel):
    """A deterministic repair recorded on the spec (re-applied after every re-layout, so repairs are never silently lost)."""
    color: str | None = None


class PosterSpec(BaseModel):
    brief_id: str
    brand: str  # brand key, e.g. "kct"
    canvas: CanvasSpec
    palette: dict[str, str] = Field(default_factory=dict)
    fonts: dict[str, str] = Field(default_factory=dict)
    background: BackgroundSpec = Field(default_factory=BackgroundSpec)
    components: list[ComponentSpec] = Field(default_factory=list)
    assets: dict[str, AssetRef] = Field(default_factory=dict)
    placed: list[PlacedElement] = Field(default_factory=list)
    mandatory: list[MandatoryItem] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    overrides: dict[str, ElementOverride] = Field(default_factory=dict)  # placed-element id -> repair
    align: Literal["center", "left"] = "center"  # composition: symmetric or left-aligned (editorial)
    flow_valign: Literal["top", "balanced", "center", "bottom"] = "balanced"  # where the content block sits in the free height
    flow_region: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # x0, y0, x1, y1 as fractions of the safe box: where flow content may sit
    template: str = "classic"  # layout template (layout/zones.py): which named zones components can be assigned to
    type_style: Literal["classic", "bold", "elegant", "editorial", "condensed"] = "classic"  # typography preset
    decorations: list[Decoration] = Field(default_factory=list)
    version: int = 0

    @model_validator(mode="after")
    def _references_resolve(self) -> "PosterSpec":
        ids = [p.id for p in self.placed]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate placed element ids")
        cids = [c.id for c in self.components]
        if len(cids) != len(set(cids)):
            raise ValueError("duplicate component ids")
        for p in self.placed:
            if p.asset_id and p.asset_id not in self.assets:
                raise ValueError(f"placed element {p.id!r} references unknown asset {p.asset_id!r}")
        for c in self.components:
            for a in c.asset_ids:
                if a not in self.assets:
                    raise ValueError(f"component {c.id!r} references unknown asset {a!r}")
        if self.background.asset_id and self.background.asset_id not in self.assets:
            raise ValueError(f"background references unknown asset {self.background.asset_id!r}")
        return self
