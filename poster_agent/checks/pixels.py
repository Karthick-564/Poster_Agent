"""Pixel measurements on the rendered PNG: WCAG contrast under text, solid-field test around logos, logo fidelity."""
from __future__ import annotations

import numpy as np
from PIL import Image

from poster_agent.spec import Box


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rel_luminance(rgb) -> float:
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2])


def contrast_ratio(l1: float, l2: float) -> float:
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def pixel_box(box: Box, scale: float, size: tuple[int, int], pad: float = 0.0) -> tuple[int, int, int, int]:
    w, h = size
    return (max(int((box.x - pad) * scale), 0), max(int((box.y - pad) * scale), 0),
            min(int((box.right + pad) * scale + 0.999), w), min(int((box.bottom + pad) * scale + 0.999), h))


def lum_array(arr: np.ndarray) -> np.ndarray:
    c = arr.astype(np.float64) / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def bg_luminances(img: Image.Image, box: Box, scale: float, text_rgb: tuple[int, int, int]) -> np.ndarray | None:
    """Relative luminance of the pixels behind the text in `box` (pixels unlike the text colour); None if the box is empty."""
    x0, y0, x1, y1 = pixel_box(box, scale, img.size)
    if x1 <= x0 or y1 <= y0:
        return None
    px = np.array(img.convert("RGB").crop((x0, y0, x1, y1))).reshape(-1, 3).astype(np.float64)
    dist = np.linalg.norm(px - np.array(text_rgb, dtype=np.float64), axis=1)
    bg = px[dist > 60]
    if len(bg) < 0.1 * len(px):
        bg = px[dist > 30]
    if len(bg) < 20:
        bg = px
    return lum_array(bg)


def text_color_ratios(img: Image.Image, box: Box, scale: float, current_rgb: tuple[int, int, int],
                      candidates: list[str]) -> list[tuple[str, float]] | None:
    """Worst-case contrast of each candidate colour (in the given order) against what is behind the text now."""
    lum = bg_luminances(img, box, scale, current_rgb)
    if lum is None:
        return None
    lo, hi = float(np.percentile(lum, 10)), float(np.percentile(lum, 90))
    return [(c, min(contrast_ratio(rel_luminance(hex_rgb(c)), lo), contrast_ratio(rel_luminance(hex_rgb(c)), hi))) for c in candidates]


def best_text_color(img: Image.Image, box: Box, scale: float, current_rgb: tuple[int, int, int],
                    candidates: list[str]) -> tuple[str, float] | None:
    """Of `candidates` (#hex), the one with the best worst-case contrast against what is behind the text now."""
    lum = bg_luminances(img, box, scale, current_rgb)
    if lum is None:
        return None
    lo, hi = float(np.percentile(lum, 10)), float(np.percentile(lum, 90))
    scored = []
    for c in candidates:
        lt = rel_luminance(hex_rgb(c))
        scored.append((min(contrast_ratio(lt, lo), contrast_ratio(lt, hi)), c))
    ratio, color = max(scored)
    return color, ratio


def text_contrast(img: Image.Image, box: Box, scale: float, text_rgb: tuple[int, int, int]) -> dict:
    """Contrast of the intended text colour against the pixels *behind* the text in `box`.

    Background pixels = those clearly unlike the text colour (anti-aliased edge pixels are ignored). The ratio is taken
    against the 10th/90th percentile background luminance, so a gradient or busy patch fails where it is worst.
    """
    x0, y0, x1, y1 = pixel_box(box, scale, img.size)
    if x1 <= x0 or y1 <= y0:
        return {"ratio": None, "bg_std": None, "bg_pixels": 0}
    px = np.array(img.convert("RGB").crop((x0, y0, x1, y1))).reshape(-1, 3).astype(np.float64)
    dist = np.linalg.norm(px - np.array(text_rgb, dtype=np.float64), axis=1)
    bg = px[dist > 60]
    if len(bg) < 0.1 * len(px):  # text fills the box: fall back to everything not text-coloured
        bg = px[dist > 30]
    if len(bg) < 20:
        # Nearly every pixel is text-coloured: the text is the same colour as what is behind it (invisible). Measure against
        # everything so this is reported as ~1:1 instead of being silently skipped.
        bg = px
    lum = lum_array(bg)
    lt = rel_luminance(text_rgb)
    lo, hi = float(np.percentile(lum, 10)), float(np.percentile(lum, 90))
    return {"ratio": min(contrast_ratio(lt, lo), contrast_ratio(lt, hi)), "bg_std": float(lum.std()),
            "bg_pixels": int(len(bg)), "bg_median_lum": float(np.median(lum))}


def ring_strips(box: Box, clear: float, canvas_w: float, canvas_h: float, gap: float = 2.0) -> list[Box]:
    """Four strips of width `clear` around `box`, starting `gap` pt off its edge so the logo's own anti-aliased edge pixels
    are not sampled (clipped to the canvas; empty ones dropped)."""
    strips = [Box(x=box.x, y=box.y - gap - clear, w=box.w, h=clear), Box(x=box.x, y=box.bottom + gap, w=box.w, h=clear),
              Box(x=box.x - gap - clear, y=box.y, w=clear, h=box.h), Box(x=box.right + gap, y=box.y, w=clear, h=box.h)]
    full = Box(x=0, y=0, w=canvas_w, h=canvas_h)
    out = []
    for s in strips:
        x0, y0, x1, y1 = max(s.x, 0), max(s.y, 0), min(s.right, full.right), min(s.bottom, full.bottom)
        if x1 - x0 > 2 and y1 - y0 > 2:
            out.append(Box(x=x0, y=y0, w=x1 - x0, h=y1 - y0))
    return out


def field_solidity(img: Image.Image, strips: list[Box], scale: float) -> dict:
    """Is the area around a logo a flat colour? Returns the worst strip std and the spread of strip means (luminance 0-255)."""
    means, stds = [], []
    for s in strips:
        x0, y0, x1, y1 = pixel_box(s, scale, img.size)
        g = np.array(img.convert("L").crop((x0, y0, x1, y1)), dtype=np.float64)
        if g.size:
            lo, hi = np.percentile(g, [2, 98])
            core = g[(g >= lo) & (g <= hi)]  # ignore stray outlier pixels
            means.append(float(core.mean()))
            stds.append(float(core.std()))
    if not means:
        return {"max_std": 0.0, "mean_spread": 0.0, "mean_lum": 255.0}
    return {"max_std": max(stds), "mean_spread": max(means) - min(means), "mean_lum": float(np.mean(means))}


def logo_correlation(img: Image.Image, box: Box, scale: float, artwork: Image.Image, size: int = 96) -> float:
    """Normalised correlation between the rendered logo region and the original artwork (1.0 = identical shape)."""
    x0, y0, x1, y1 = pixel_box(box, scale, img.size)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    rendered = img.convert("L").crop((x0, y0, x1, y1)).resize((size, size), Image.LANCZOS)
    art = artwork
    if art.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", art.size, (255, 255, 255, 255))
        bg.alpha_composite(art.convert("RGBA"))
        art = bg
    art = art.convert("L").resize((size, size), Image.LANCZOS)
    a, b = np.asarray(rendered, dtype=np.float64).ravel(), np.asarray(art, dtype=np.float64).ravel()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
