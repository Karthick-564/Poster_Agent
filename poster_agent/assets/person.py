"""Reframe a supplied photo of a person (head, head and shoulders, half body) by CROPPING it. The photo is never generated or altered by a
model: the pixels inside the crop are the original pixels; only the framing changes (guardrail 4).

How the head is found (reported in `Framing.method`):
- `yunet`: OpenCV's YuNet face detector, when its ONNX file is at data/models/face_detection_yunet_2023mar.onnx (download once from
  opencv_zoo; the largest face is used).
- `silhouette`: without that file, from the person's matte (rembg, local): the head is the first bulge below the top of the silhouette,
  before the neck narrows and the shoulders widen. Works for a single person; a group photo needs YuNet.
- `none`: nothing usable was found: the photo is returned centred and uncropped and `Framing.note` says so.

The result says what was actually possible: `limited` is True when the photo does not extend as far as the preset asks (for example a tight
headshot cannot become head-and-shoulders), because a crop can never add pixels that are not in the photo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from poster_agent.spec.brand import ROOT

YUNET = ROOT / "data" / "models" / "face_detection_yunet_2023mar.onnx"

# preset: (crop width in head-widths, headroom above the head in head-heights, space below the head in head-heights, default aspect w/h)
PRESETS: dict[str, tuple[float, float, float, float]] = {
    "face": (1.62, 0.13, 0.2, 1.0),  # just the face (hair to chin, a little neck room), square: made for a circle
    "head": (1.9, 0.25, 0.25, 1.0),
    "head_and_shoulders": (2.9, 0.22, 1.05, 0.8),
    "half_body": (4.0, 0.2, 2.6, 0.75),
}


@dataclass
class Head:
    x: float
    y: float
    w: float
    h: float


@dataclass
class Framing:
    box: tuple[int, int, int, int]  # left, top, right, bottom of the crop in the source photo
    method: str
    limited: bool
    note: str
    head: Head | None = None


def head_from_mask(mask: np.ndarray) -> Head | None:
    """The head of the single person in a boolean matte, from the width profile of the silhouette: it widens to the head, narrows at the
    neck, widens again at the shoulders. A silhouette with no neck (a tight headshot) is returned whole."""
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) < 8:
        return None
    top, bottom = int(rows[0]), int(rows[-1])
    height = bottom - top + 1
    width = np.array([mask[r].sum() for r in range(top, bottom + 1)], dtype=float)
    k = max(height // 50, 1)
    sm = np.convolve(width, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    limit = max(int(0.6 * height), 4)  # the head is within the upper 60% of the person
    neck, peak_val, peak_r = None, 0.0, 0
    for r in range(limit):  # walk down from the top: widest so far = the head; the first clear narrowing after it = the neck
        if sm[r] > peak_val:
            peak_val, peak_r = sm[r], r
        elif r > peak_r and sm[r] < 0.82 * peak_val:
            neck = r
            break
    if neck is None:
        end = height  # no neck found: the whole silhouette is the head (a tight crop)
    else:
        end = neck
        while end + 1 < limit and sm[end + 1] <= sm[end]:  # to the narrowest row of the neck
            end += 1
    ys = slice(top, top + max(end, 1))
    cols = np.where(mask[ys].any(axis=0))[0]
    if len(cols) == 0:
        return None
    return Head(x=float(cols[0]), y=float(top), w=float(cols[-1] - cols[0] + 1), h=float(max(end, 1)))


def head_from_yunet(img: Image.Image) -> Head | None:
    """Largest face from YuNet, widened to include the hair (None if the model file is missing or finds nothing)."""
    if not YUNET.exists():
        return None
    import cv2
    bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    det = cv2.FaceDetectorYN.create(str(YUNET), "", (bgr.shape[1], bgr.shape[0]), 0.7)
    _, faces = det.detect(bgr)
    if faces is None or len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])[:4]
    return Head(x=float(x - 0.1 * w), y=float(y - 0.35 * h), w=float(1.2 * w), h=float(1.55 * h))  # face box -> head incl. hair


def plan_crop(size: tuple[int, int], head: Head, preset: str, aspect: float | None = None) -> tuple[tuple[int, int, int, int], bool]:
    """Crop box (left, top, right, bottom) for `preset` around `head`, kept inside the photo; (box, limited)."""
    cw_f, above, below, default_aspect = PRESETS[preset]
    aspect = aspect or default_aspect
    W, H = size
    cw = cw_f * head.w
    top = head.y - above * head.h
    bottom = head.y + head.h + below * head.h
    ch = bottom - top
    if cw / ch < aspect:  # too narrow for the aspect: widen
        cw = ch * aspect
    else:  # too wide: add height below (never above the headroom)
        bottom = top + cw / aspect
        ch = bottom - top
    cx = head.x + head.w / 2
    left, right = cx - cw / 2, cx + cw / 2
    limited = left < 0 or top < 0 or right > W or bottom > H
    if limited:  # keep the aspect; shrink to what the photo has, keeping the head's place
        s = min(1.0, W / cw, H / ch, (min(cx, W - cx) * 2) / cw if min(cx, W - cx) > 0 else 1.0)
        cw, ch = cw * s, ch * s
        left, right = min(max(cx - cw / 2, 0), W - cw), min(max(cx - cw / 2, 0), W - cw) + cw
        top = min(max(top, 0), H - ch)
        bottom = top + ch
    return (int(round(left)), int(round(top)), int(round(right)), int(round(bottom))), limited


def frame_person(img: Image.Image, preset: str = "head_and_shoulders", aspect: float | None = None, mask: np.ndarray | None = None
                 ) -> tuple[Image.Image, Framing]:
    """Crop `img` to the preset framing. `mask` (bool matte of the person) is computed with rembg when not given and no face is found."""
    if preset not in PRESETS:
        raise ValueError(f"unknown framing {preset!r}; use one of {sorted(PRESETS)}")
    img = img.convert("RGB")
    head, method = head_from_yunet(img), "yunet"
    if head is None:
        if mask is None:
            from .matting import subject_mask  # local rembg; the matte only LOCATES the person, the pixels cropped are the photo's own
            mask = subject_mask(img)
        head, method = head_from_mask(mask), "silhouette"
    if head is None:
        side = min(img.size)
        box = ((img.width - side) // 2, (img.height - side) // 2, (img.width + side) // 2, (img.height + side) // 2)
        fr = Framing(box, "none", True, "no person found; the photo was centre-cropped, not reframed")
        return img.crop(box), fr
    box, limited = plan_crop(img.size, head, preset, aspect)
    note = (f"{preset} framing from {method}" + ("; the photo does not extend that far, so the crop is as wide as the photo allows"
                                                  if limited else ""))
    return img.crop(box), Framing(box, method, limited, note, head)


def apply_framing(spec, out_dir: Path) -> list[str]:
    """Swap every headshot / photo element that asks for `framing` onto a reframed COPY of its supplied photo. Idempotent. Returns notes."""
    import hashlib
    from poster_agent.spec import AssetRef, ElementKind
    notes: list[str] = []
    for el in spec.placed:
        preset = el.props.get("framing")
        if el.kind != ElementKind.image or not preset or el.asset_id not in spec.assets:
            continue
        src = spec.assets[el.asset_id]
        if src.treatment.get("framing"):
            continue
        key = hashlib.sha1(f"{src.path}|{preset}".encode()).hexdigest()[:10]
        aid = f"{src.id}__{preset}_{key}"
        if aid not in spec.assets:
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / f"{aid}.png"
            with Image.open(ROOT / src.path) as im:
                cropped, fr = frame_person(im, preset)
            cropped.save(dest)
            spec.assets[aid] = AssetRef(id=aid, kind=src.kind, path=dest.resolve().relative_to(ROOT).as_posix(), provenance=src.provenance,
                                        treatment={"framing": preset, "method": fr.method, "limited": fr.limited, "source": src.path})
            notes.append(f"{el.id}: {fr.note}")
        el.asset_id = aid
    return notes
