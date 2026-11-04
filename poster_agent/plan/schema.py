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


class PlanBackground(BaseModel):
    kind: Literal["solid", "gradient", "generated"] = "solid"
    color: str = "surface"  # palette name or #hex (also the fallback ground under generated art)
    gradient: list[PlanStop] = Field(default_factory=list)
    angle: float = 90.0
    art: PlanArt | None = None  # required when kind == "generated"


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


class PlanOutput(BaseModel):
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
