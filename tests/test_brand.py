from pathlib import Path

import pytest
from PIL import ImageFont

from poster_agent.spec.brand import BRANDS_DIR, ROOT, BrandProfile, list_brands, load_brand

FONTS_DIR = BRANDS_DIR / "fonts"


def test_expected_brands_present():
    assert {"kct", "kclas", "kctbs"} <= set(list_brands())


@pytest.mark.parametrize("key", ["kct", "kclas", "kctbs"])
def test_brand_loads_and_logos_resolve(key):
    b = load_brand(key)
    assert isinstance(b, BrandProfile) and b.key == key
    assert b.logo().aspect > 0
    for logo in b.logos.values():
        assert (ROOT / logo.path).exists() and logo.aspect > 0
    assert b.logo_rules.clear_space_mm == 10 and not b.logo_rules.allow_gradient_behind
    assert "logo" in b.mandatory_elements


@pytest.mark.parametrize("key", ["kct", "kclas", "kctbs"])
def test_brand_fonts_are_shipped(key):
    """Every non-system font family a brand asks for must exist in brands/fonts (and so in VectorCraft's font dir)."""
    shipped = set()
    for f in FONTS_DIR.glob("*.ttf"):
        family, style = ImageFont.truetype(str(f), 12).getname()
        shipped.add((family, style))
    for role, font in load_brand(key).fonts.items():
        if font.family == "Arial":
            continue
        assert (font.family, font.style) in shipped, f"{key}.{role}: {font.family} {font.style} not in brands/fonts"


def test_kct_values_from_guide():
    b = load_brand("kct")
    assert b.palette["primary"] == "#1e2842" and b.palette["accent"] == "#85cebd"
    assert b.fonts["alt"].substitute_for == "Helvetica Neue LT Std"
    assert b.logo_rules.partner_logo_side == "left"


def test_unknown_brand_and_bad_palette(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_brand("nope")
    with pytest.raises(ValueError):
        BrandProfile.model_validate({
            "key": "x", "name": "X", "short_name": "X", "palette": {"primary": "blue", "surface": "#fff", "text": "#000"},
            "fonts": {"display": {"family": "A"}, "body": {"family": "A"}},
            "logos": {"l": {"file": "l.png", "kind": "primary"}}, "default_logo": "l"})
