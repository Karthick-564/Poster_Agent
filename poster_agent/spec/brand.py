"""BrandProfile: per-institution rules loaded from brands/<key>/brand.yaml (guardrail 5: behaviour comes from here)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from PIL import Image
from pydantic import BaseModel, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parents[2]
BRANDS_DIR = ROOT / "brands"
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


class FontRole(BaseModel):
    family: str
    style: str = "Regular"  # VectorCraft font style name: Regular, Bold, Light, ExtraBold...
    fallbacks: list[str] = Field(default_factory=list)
    substitute_for: str | None = None  # brand-specified commercial font this open font stands in for


class LogoVariant(BaseModel):
    file: str  # relative to brands/<key>/
    kind: Literal["primary", "secondary", "emblem"]
    language: Literal["en", "ta"] = "en"
    field: Literal["any", "light", "dark"] = "any"  # background the artwork may sit on
    min_width_pct: float = 10.0  # minimum logo width as % of canvas width
    aspect: float = 0.0  # width/height, read from the file at load time
    path: str = ""  # project-root-relative path, resolved at load time


class LogoRules(BaseModel):
    clear_space_mm: float = 10.0
    allow_gradient_behind: bool = False
    placement_preference: list[str] = Field(default_factory=lambda: ["top-right", "top-left"])
    partner_logo_side: Literal["left", "right"] = "left"  # which side of a partner logo ours sits on
    forbidden: list[str] = Field(default_factory=lambda: [
        "rotate", "non-proportional scale", "add effects/shadows/outlines", "recolour", "crop", "place on gradient"])


class FooterDefaults(BaseModel):
    website: str | None = None
    phone: str | None = None
    address: str | None = None
    email: str | None = None
    auto_include: bool = False  # only present when the brief/plan asks for it


class BrandProfile(BaseModel):
    key: str
    name: str
    short_name: str
    palette: dict[str, str]
    fonts: dict[str, FontRole]
    logos: dict[str, LogoVariant]
    default_logo: str
    logo_rules: LogoRules = Field(default_factory=LogoRules)
    mandatory_elements: list[str] = Field(default_factory=lambda: ["logo"])
    footer: FooterDefaults = Field(default_factory=FooterDefaults)
    min_text_contrast: float = 4.5
    safe_margin_pct: float = 4.5
    background_rules: list[str] = Field(default_factory=list)
    forbidden_grounds: list[str] = Field(default_factory=list)  # palette names that may not be used as a fill/background
    # Institutional posters are clean: structural rules only. The ornamental kinds (glow, dot_grid, rings, corner_shapes,
    # diagonal_band, watermark_text) still exist in layout/decor.py but a brand has to opt in to them explicitly.
    allowed_decorations: list[str] = Field(default_factory=lambda: ["border", "frame", "stripe"])
    photo_treatments: list[str] = Field(default_factory=lambda: ["duotone"])  # colour treatments the brand allows on supplied photos
    notes: list[str] = Field(default_factory=list)

    @field_validator("palette")
    @classmethod
    def _palette_hex(cls, v: dict[str, str]) -> dict[str, str]:
        for name, value in v.items():
            if not _HEX.match(value):
                raise ValueError(f"palette[{name}]={value!r} is not #rrggbb")
        for required in ("primary", "surface", "text"):
            if required not in v:
                raise ValueError(f"palette must define {required!r}")
        return {k: c.lower() for k, c in v.items()}

    @model_validator(mode="after")
    def _roles_and_default_logo(self) -> "BrandProfile":
        for role in ("display", "body"):
            if role not in self.fonts:
                raise ValueError(f"fonts must define {role!r}")
        if self.default_logo not in self.logos:
            raise ValueError(f"default_logo {self.default_logo!r} not in logos {sorted(self.logos)}")
        return self

    def logo(self, name: str | None = None) -> LogoVariant:
        return self.logos[name or self.default_logo]


def load_brand(key: str, brands_dir: Path = BRANDS_DIR) -> BrandProfile:
    key = key.lower()
    brand_dir = brands_dir / key
    cfg = brand_dir / "brand.yaml"
    if not cfg.exists():
        known = sorted(p.parent.name for p in brands_dir.glob("*/brand.yaml"))
        raise FileNotFoundError(f"no brand profile for {key!r} (known: {known})")
    profile = BrandProfile.model_validate(yaml.safe_load(cfg.read_text(encoding="utf-8")))
    for name, logo in profile.logos.items():
        file = brand_dir / logo.file
        if not file.exists():
            raise FileNotFoundError(f"brand {key!r} logo {name!r}: missing file {file}")
        with Image.open(file) as im:
            logo.aspect = im.width / im.height
        try:
            logo.path = file.relative_to(ROOT).as_posix()
        except ValueError:
            logo.path = file.as_posix()
    return profile


def list_brands(brands_dir: Path = BRANDS_DIR) -> list[str]:
    return sorted(p.parent.name for p in brands_dir.glob("*/brand.yaml"))
