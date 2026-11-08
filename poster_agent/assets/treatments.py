"""Brand colour treatments for supplied photos (guardrail 4: never diffusion; colour-treat only as brand rules allow).

`duotone` maps a photo's luminance onto a two-colour ramp (shadows -> a dark brand colour, highlights -> a light one), the way
institutional posters unify mixed event photos. It is a deterministic pixel operation on a COPY of the file: the supplied original is
never modified, the treated copy is recorded on the asset (`treatment`), and logos and headshots are never treated.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

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


# ---- generated backgrounds are a quiet backdrop, not the subject ------------------------------------------------------------------
def art_energy(img: Image.Image) -> dict[str, float]:
    """How loud a picture is: mean saturation, luminance contrast and local detail (all 0..1)."""
    small = img.convert("RGB").resize((128, 128))
    sat = float((np.asarray(small.convert("HSV"), dtype=float)[..., 1] / 255.0).mean())
    lum = np.asarray(small.convert("L"), dtype=float) / 255.0
    gy, gx = np.gradient(lum)
    return {"saturation": round(sat, 3), "contrast": round(float(lum.std()), 3), "detail": round(float(np.hypot(gx, gy).mean()), 4)}


def calm_strengths(e: dict[str, float]) -> dict[str, float]:
    """How much to tone down each aspect of a picture, 0..1, from how far THAT aspect exceeds what text can sit on. Saturation, contrast
    and detail are judged separately: a pale picture with fine detail is blurred a little, not drained of its colour."""
    clip = lambda v: float(min(max(v, 0.0), 1.0))
    return {"saturation": clip((e["saturation"] - 0.28) / 0.30), "contrast": clip((e["contrast"] - 0.16) / 0.20),
            "detail": clip((e["detail"] - 0.015) / 0.030)}


def calm_strength(e: dict[str, float]) -> float:
    """One number for reporting: the loudest aspect, never below 0.25 (every backdrop is nudged a little toward the brand tint)."""
    return max(0.25, *calm_strengths(e).values())


def calm_art(img: Image.Image, tint_hex: str, strength: float | None = None) -> tuple[Image.Image, dict]:
    """Tone generated art down so type can sit on it: lower saturation, compressed contrast, a touch of blur (depth of field) and a pull
    toward the brand tint, each in proportion to how loud that aspect is. Deterministic; applies to generated art only (supplied photos
    are never altered this way). `strength` forces one value for all aspects (tests, manual use)."""
    before = art_energy(img)
    k = calm_strengths(before) if strength is None else {"saturation": strength, "contrast": strength, "detail": strength}
    overall = calm_strength(before) if strength is None else strength
    rgb = img.convert("RGB")
    h, s, v = rgb.convert("HSV").split()
    s = s.point(lambda p: int(p * (1 - 0.55 * k["saturation"])))
    out = np.asarray(Image.merge("HSV", (h, s, v)).convert("RGB"), dtype=float)
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = mean + (out - mean) * (1 - 0.40 * k["contrast"])  # compress contrast around the picture's own mean
    mix = 0.08 + 0.22 * max(k["saturation"], k["contrast"], 0.0)  # pull toward the brand tint follows the colour / contrast calming
    tint = np.array([int(tint_hex[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    out = out * (1 - mix) + tint * mix
    res = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    if k["detail"] > 0.1:
        res = res.filter(ImageFilter.GaussianBlur(radius=max(res.width / 1080.0, 1.0) * 1.8 * k["detail"]))
    return res, {"strength": round(overall, 2), "aspects": {a: round(x, 2) for a, x in k.items()}, "before": before, "after": art_energy(res)}


def calm_under_text(img: Image.Image, boxes: list, width: float, height: float, wash_hex: str, target: float
                    ) -> tuple[Image.Image, dict]:
    """Soften the picture ONLY where text sits: a feathered blur plus a wash of `wash_hex` inside the (padded) text boxes, repeated
    (stronger each pass) until the fine detail under the text is at or below `target`. The rest of the picture is untouched, so it keeps
    its full colour and contrast. Deterministic; generated art only."""
    from poster_agent.analysis import text_detail
    rgb = img.convert("RGB")
    sx, sy = rgb.width / width, rgb.height / height
    pad = 0.025 * width
    mask = Image.new("L", rgb.size, 0)
    from PIL import ImageDraw
    d = ImageDraw.Draw(mask)
    for b in boxes:
        d.rounded_rectangle(((b.x - pad) * sx, (b.y - pad) * sy, (b.right + pad) * sx, (b.bottom + pad) * sy), radius=pad * sx, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=0.5 * pad * sx))  # feathered: no visible edge
    wash = Image.new("RGB", rgb.size, tuple(int(wash_hex[i:i + 2], 16) for i in (1, 3, 5)))
    before = text_detail(rgb, boxes, width, height)
    out, passes = rgb, 0
    for p in range(3):
        if text_detail(out, boxes, width, height) <= target:
            break
        passes = p + 1
        soft = out.filter(ImageFilter.GaussianBlur(radius=rgb.width / 1080.0 * 3.0 * (p + 1)))
        soft = Image.blend(soft, wash, 0.22 + 0.12 * p)
        out = Image.composite(soft, out, mask)
    return out, {"before": round(before, 4), "after": round(text_detail(out, boxes, width, height), 4), "passes": passes, "target": target}


def calm_background(path: Path, brand: BrandProfile, tone: str) -> str:
    """Calm the generated background file in place (the untouched original is kept as <name>_raw.png). Returns a one-line note."""
    path = Path(path)
    raw = path.with_name(path.stem + "_raw" + path.suffix)
    shutil.copy(path, raw)  # `path` was just (re)generated: keep that untouched original beside the calmed file
    with Image.open(raw) as im:
        tint = resolve_color(brand, "primary" if tone == "dark" else "surface")
        out, info = calm_art(im, tint)
    out.save(path)
    b, a = info["before"], info["after"]
    return (f"background art calmed (strength {info['strength']:.2f}: saturation {b['saturation']:.2f}->{a['saturation']:.2f}, "
            f"contrast {b['contrast']:.2f}->{a['contrast']:.2f}, detail {b['detail']:.3f}->{a['detail']:.3f})")
