"""Local background removal (rembg) -> clean RGBA cutouts. Runs on this PC's CPU, so it needs nothing from the GPU box
(the ComfyUI install has the RemoveBackground node but no matting model file)."""
from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

MODEL = "isnet-general-use"


@lru_cache(maxsize=1)
def _session(model: str = MODEL):
    from rembg import new_session
    return new_session(model)


def cutout(src: Image.Image | Path | bytes, *, feather: float = 0.8, erode_px: int = 1, pad_pct: float = 0.02) -> Image.Image:
    """Matte the subject: returns RGBA cropped to the subject (+padding) with a clean, slightly feathered edge."""
    from rembg import remove
    if isinstance(src, (bytes, bytearray)):
        src = Image.open(BytesIO(src))
    elif isinstance(src, Path):
        src = Image.open(src)
    rgba = remove(src.convert("RGB"), session=_session()).convert("RGBA")
    alpha = rgba.getchannel("A")
    a = np.array(alpha)
    a[a < 12] = 0  # drop faint stray alpha
    alpha = Image.fromarray(a)
    if erode_px:
        alpha = alpha.filter(ImageFilter.MinFilter(2 * erode_px + 1))  # shave the halo of background colour at the edge
    if feather:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather))
    rgba.putalpha(alpha)
    bbox = alpha.point(lambda v: 255 if v > 24 else 0).getbbox()
    if not bbox:
        raise ValueError("matting found no subject")
    pad = int(pad_pct * max(rgba.size))
    box = (max(bbox[0] - pad, 0), max(bbox[1] - pad, 0), min(bbox[2] + pad, rgba.width), min(bbox[3] + pad, rgba.height))
    return rgba.crop(box)


def alpha_coverage(img: Image.Image) -> float:
    """Fraction of pixels that are mostly opaque (a sanity metric: ~0 or ~1 means the matting failed)."""
    a = np.array(img.convert("RGBA").getchannel("A"))
    return float((a > 128).mean())


def subject_mask(img: Image.Image) -> np.ndarray:
    """Boolean matte of the main subject (used to LOCATE a person, never to alter the photo)."""
    from rembg import remove
    return np.array(remove(img.convert("RGB"), session=_session(), only_mask=True)) > 128
