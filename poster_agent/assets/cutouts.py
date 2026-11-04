"""A supplied photo of a person (or object) turned into a cutout: only a MATTE is computed (rembg, local) and applied as transparency; the
photo's own pixels are untouched, nothing is repainted or generated (guardrail 4). The cutout is a COPY of the supplied file."""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from poster_agent.spec import AssetKind, AssetRef, ElementKind
from poster_agent.spec.brand import ROOT

CUTTABLE = {AssetKind.headshot, AssetKind.photo}  # supplied photos; generated heroes are already cutouts


def apply_cutouts(spec, out_dir: Path, matte=None) -> list[str]:
    """Swap every image element that asks for `cutout` onto a matted copy of its supplied photo. Idempotent. Returns notes (what happened,
    including when a matte looked wrong and the photo was left as supplied)."""
    from .matting import alpha_coverage, cutout
    matte = matte or cutout
    notes: list[str] = []
    for el in spec.placed:
        if el.kind != ElementKind.image or not el.props.get("cutout") or el.asset_id not in spec.assets:
            continue
        src = spec.assets[el.asset_id]
        if src.kind not in CUTTABLE or src.treatment.get("cutout"):
            continue
        key = hashlib.sha1(f"{src.path}|cutout".encode()).hexdigest()[:10]
        aid = f"{src.id}__cutout_{key}"
        if aid not in spec.assets:
            with Image.open(ROOT / src.path) as im:
                rgba = matte(im.convert("RGB"), pad_pct=0.0)
            cov = alpha_coverage(rgba)
            if not 0.08 < cov < 0.97:  # a matte that kept (almost) everything or (almost) nothing failed: do not use it
                notes.append(f"{el.id}: the background could not be separated cleanly (kept {cov:.0%}); photo left as supplied")
                el.props.pop("cutout", None)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / f"{aid}.png"
            rgba.save(dest)
            spec.assets[aid] = AssetRef(id=aid, kind=src.kind, path=dest.resolve().relative_to(ROOT).as_posix(),
                                        provenance=src.provenance, treatment={"cutout": True, "source": src.path})
            notes.append(f"{el.id}: background of the supplied photo removed (matte only, pixels unchanged), kept {cov:.0%}")
        el.asset_id = aid
    return notes
