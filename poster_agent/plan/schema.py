"""What the LLM planner returns. Component specs only — never coordinates (guardrail: the LLM chooses, the solver places)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from poster_agent.spec import ComponentSpec


class PlanStop(BaseModel):
    offset: float = Field(ge=0.0, le=1.0)
    color: str  # palette name or #hex


class PlanArt(BaseModel):
    """Art direction for a generated background. Imagery only — never text, logos or people."""
    description: str
    style: str = ""
    lighting: str = ""
    medium: str = "digital illustration"
    tone: Literal["light", "dark"] = "light"  # which text colour the art must support (dark art -> light text)
    palette: list[str] = Field(default_factory=list)  # palette names or #hex
    subject: str = ""  # the one focal subject and where it sits in the frame ("a robot arm in the lower right third")
    composition: str = ""  # camera angle, depth layers (foreground / middle / background), where the quiet space is
    devices: str = ""  # graphic structure that makes it designable: curved horizon, diagonal light beam, layered shapes, a calm panel
    color_roles: str = ""  # which colour dominates, which accents, where the light comes from


class PlanPanel(BaseModel):
    """A two-tone ground: a solid colour block from just above the component `starts_at` down to the bottom edge, over the ground."""
    color: str  # palette name or #hex
    starts_at: str  # id of the flow component the block begins at (it and everything below it sit on the block)


class PlanBackground(BaseModel):
    kind: Literal["solid", "gradient", "generated"] = "solid"
    color: str = "surface"  # palette name or #hex (also the fallback ground under generated art)
    gradient: list[PlanStop] = Field(default_factory=list)
    angle: float = 90.0
    art: PlanArt | None = None  # required when kind == "generated"
    panel: PlanPanel | None = None  # optional two-tone ground (solid / gradient grounds only)


class PlanGeneratedAsset(BaseModel):
    """An object hero the agent generates (Qwen) and cuts out. Never a logo, photo of a person, or anything with text."""
    id: str
    kind: Literal["hero_cutout"] = "hero_cutout"
    subject: str  # what the object is, in plain descriptive prose
    aspect: Literal["1:1", "4:5", "3:4", "16:9", "9:16"] = "1:1"


class PlanDecoration(BaseModel):
    """A brand-native decorative layer behind the content (see BRAND.allowed_decorations)."""
    kind: str
    region: str = "full"  # full | top | bottom | left | right | center
    color: str = "accent"  # palette name or #hex
    opacity: float = 0.12
    text: str | None = None  # watermark_text only: verbatim from the brief (e.g. the year)
    density: float = 1.0


class PlanConcept(BaseModel):
    """The design decision, made BEFORE the layout and checked against it."""
    idea: str  # the one idea of this poster, one sentence
    focal_point: str  # id of the component the eye lands on first (it must have emphasis "hero")
    reading_order: list[str]  # component ids in the order the eye should read them
    devices: list[str]  # design devices used, from plan.validate.DEVICES
    contrast: str  # the main contrast that creates hierarchy: scale, colour, weight or space


class PlanOutput(BaseModel):
    concept: PlanConcept | None = None  # required from the planner; decided first
    format: str  # story | fourfive | square | a3 | a4
    background: PlanBackground = Field(default_factory=PlanBackground)
    components: list[ComponentSpec]
    generated_assets: list[PlanGeneratedAsset] = Field(default_factory=list)
    align: Literal["center", "left"] = "center"  # composition: symmetric, or left-aligned stacks with a split layout
    flow_valign: Literal["top", "balanced", "center", "bottom"] = "balanced"
    template: str = "classic"  # layout template with named zones; components then carry a `zone`
    look: str = ""  # set by the system (plan/looks.py), not by the planner
    type_style: Literal["classic", "bold", "elegant", "editorial", "condensed"] = "classic"
    decorations: list[PlanDecoration] = Field(default_factory=list)
    rationale: str = ""  # one or two sentences on the design choice (logged, never printed)
