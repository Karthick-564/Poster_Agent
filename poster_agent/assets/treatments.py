"""Brand colour treatments for supplied photos (guardrail 4: never diffusion; colour-treat only as brand rules allow).

`duotone` maps a photo's luminance onto a two-colour ramp (shadows -> a dark brand colour, highlights -> a light one), the way
institutional posters unify mixed event photos. It is a deterministic pixel operation on a COPY of the file: the supplied original is
never modified, the treated copy is recorded on the asset (`treatment`), and logos and headshots are never treated.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageOps

from poster_agent.layout.base import resolve_color
from poster_agent.spec import AssetKind, AssetRef, ElementKind, PosterSpec
from poster_agent.spec.brand import ROOT, BrandProfile

TREATABLE_ROLES = {"photo", "hero"}  # never logos, partner logos or headshots
TREATABLE_KINDS = {AssetKind.photo, AssetKind.texture}  # generated art and cutouts are already ours; people photos stay as supplied


def duotone(img: Image.Image, dark: str, light: str) -> Image.Image:
    """Luminance -> dark..light ramp. Alpha (if any) is preserved."""
    rgba = img.convert("RGBA")
    grey = ImageOps.autocontrast(rgba.convert("L"), cutoff=1)
    out = ImageOps.colorize(grey, black=dark, white=light).convert("RGBA")
    out.putalpha(rgba.getchannel("A"))
    return out


def apply_treatments(spec: PosterSpec, brand: BrandProfile, out_dir: Path) -> list[str]:
    """Swap every image element that asks for a treatment onto a treated copy of its asset. Idempotent. Returns notes."""
    notes: list[str] = []
    for el in spec.placed:
        mode = el.props.get("treatment")
        if el.kind != ElementKind.image or not mode or el.asset_id not in spec.assets:
            continue
        src = spec.assets[el.asset_id]
        if mode not in brand.photo_treatments:
            notes.append(f"{el.id}: treatment {mode!r} is not allowed by the {brand.short_name} brand rules; photo left as supplied")
            continue
        if el.role not in TREATABLE_ROLES or src.kind not in TREATABLE_KINDS or src.treatment.get("tone"):
            if not src.treatment.get("tone"):
                notes.append(f"{el.id}: {el.role}/{src.kind.value} is never colour-treated; left as supplied")
            continue
        dark_name, light_name = el.props.get("duotone", ["primary", "surface"])
        dark, light = resolve_color(brand, dark_name), resolve_color(brand, light_name)
        key = hashlib.sha1(f"{src.path}|{mode}|{dark}|{light}".encode()).hexdigest()[:10]
        aid = f"{src.id}__{mode}_{key}"
        if aid not in spec.assets:
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / f"{aid}.png"
            with Image.open(ROOT / src.path) as im:
                duotone(im, dark, light).save(dest)
            spec.assets[aid] = AssetRef(id=aid, kind=src.kind, path=dest.resolve().relative_to(ROOT).as_posix(),
                                        provenance=src.provenance, treatment={"tone": mode, "dark": dark, "light": light, "source": src.path})
        el.asset_id = aid
        notes.append(f"{el.id}: {mode} {dark}->{light} applied to a copy of {src.id}")
    return notes
