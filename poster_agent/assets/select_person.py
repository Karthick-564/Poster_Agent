"""Pick one person out of a supplied photo from a plain description ("the man on the right", "woman in the red saree") and crop to them.

The local vision model (Qwen3-VL on the VLM endpoint, the same one the critic uses) is asked for the person's bounding box; it understands
relations and clothing, which a plain detector does not. Qwen3-VL answers in JSON `bbox_2d` on a 0-1000 grid. The crop is the photo's own
pixels (guardrail 4); the box only locates the person. Pass the result to `frame_person` for head / shoulder framing.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from PIL import Image

SYSTEM = ("You locate people in photos. Reply with JSON only: a list with one object {\"bbox_2d\": [x1, y1, x2, y2], \"label\": str} "
          "for the single person that matches the description, using a 0-1000 coordinate grid (0,0 top-left). If nobody matches, reply [].")


def parse_box(raw, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """First usable box in a model reply (JSON, possibly fenced or truncated) as pixel (left, top, right, bottom); None if there is none."""
    text = raw if isinstance(raw, str) else json.dumps(raw)
    nums = None
    try:
        data = json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip())
        data = data if isinstance(data, list) else data.get("boxes", [data]) if isinstance(data, dict) else []
        nums = next((d["bbox_2d"] for d in data if isinstance(d, dict) and len(d.get("bbox_2d", [])) == 4), None)
    except (ValueError, AttributeError):
        pass
    if nums is None:  # truncated or chatty reply: take the first four numbers after bbox_2d
        m = re.search(r"bbox_2d\"?\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", text)
        nums = [float(g) for g in m.groups()] if m else None
    if nums is None:
        return None
    W, H = size
    x1, y1, x2, y2 = (float(n) for n in nums)
    box = (int(max(x1, 0) / 1000 * W), int(max(y1, 0) / 1000 * H), int(min(x2, 1000) / 1000 * W), int(min(y2, 1000) / 1000 * H))
    return box if box[2] - box[0] > 8 and box[3] - box[1] > 8 else None


def select_person(img: Image.Image, description: str, vlm, pad: float = 0.12) -> tuple[Image.Image, dict]:
    """Crop `img` to the person matching `description`. Returns (crop, info). If the model finds nobody, the photo is returned unchanged and
    info["found"] is False (the caller must say so; nothing is guessed)."""
    img = img.convert("RGB")
    small = img.copy()
    small.thumbnail((1280, 1280))
    with tempfile.TemporaryDirectory() as td:  # the client reads image files
        p = Path(td) / "photo.png"
        small.save(p)
        raw = vlm.chat(SYSTEM, f"Description: {description}", images=[p], temperature=0.0)
    box = parse_box(raw, img.size)
    if box is None:
        return img, {"found": False, "description": description, "note": "the vision model found no matching person; photo left as supplied"}
    l, t, r, b = box
    px, py = int((r - l) * pad), int((b - t) * pad)
    crop_box = (max(l - px, 0), max(t - py, 0), min(r + px, img.width), min(b + py, img.height))
    return img.crop(crop_box), {"found": True, "description": description, "box": crop_box, "model": getattr(vlm, "model", "vlm")}


def select_person_file(src: Path, dest: Path, description: str, vlm) -> dict:
    with Image.open(src) as im:
        out, info = select_person(im, description, vlm)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.save(dest)
    return info
