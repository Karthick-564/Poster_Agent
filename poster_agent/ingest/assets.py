"""Asset inventory: the files that arrive with a brief, each with an id, a kind and a label the planner can reference.

Kinds come from a filename convention (`<kind>__<label>.<ext>`, e.g. `guest__dr_ramarao.jpg`, `partner__tancoir.png`);
files without a prefix are treated as event photos. We never identify people from their faces — labels come from people.
"""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

from poster_agent.spec import AssetKind, AssetRef
from poster_agent.spec.brand import ROOT

KIND_PREFIXES = {
    "guest": AssetKind.headshot, "speaker": AssetKind.headshot, "headshot": AssetKind.headshot,
    "partner": AssetKind.partner_logo, "sponsor": AssetKind.partner_logo,
    "photo": AssetKind.photo, "event": AssetKind.photo, "texture": AssetKind.texture,
}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
MIN_SIDE_PX = 600  # below this a photo will look soft at poster size; reported as a warning


class InventoryItem(BaseModel):
    id: str
    path: str  # project-root-relative if inside the project
    kind: AssetKind
    label: str
    width: int
    height: int
    warnings: list[str] = []

    def to_asset(self) -> AssetRef:
        return AssetRef(id=self.id, kind=self.kind, path=self.path, provenance="supplied")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def scan_assets(folder: Path) -> dict[str, InventoryItem]:
    """Build the inventory from a folder of supplied files."""
    items: dict[str, InventoryItem] = {}
    for f in sorted(Path(folder).iterdir()):
        if f.suffix.lower() not in IMAGE_EXT:
            continue
        prefix, _, rest = f.stem.partition("__")
        kind = KIND_PREFIXES.get(prefix.lower()) if rest else None
        label = rest if rest else f.stem
        kind = kind or AssetKind.photo
        with Image.open(f) as im:
            w, h = im.size
        warnings = []
        if kind in (AssetKind.headshot, AssetKind.photo) and min(w, h) < MIN_SIDE_PX:
            warnings.append(f"low resolution {w}x{h} (< {MIN_SIDE_PX}px on the short side)")
        aid = _slug(f.stem)
        try:
            path = f.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            path = f.resolve().as_posix()
        items[aid] = InventoryItem(id=aid, path=path, kind=kind, label=label.replace("_", " "), width=w, height=h,
                                   warnings=warnings)
    return items
