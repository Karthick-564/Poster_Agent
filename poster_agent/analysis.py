"""Look at the art before placing text on it: where it is busy, where its subject is, where it is calm.

`analyse` turns the background raster (as the compiler draws it: cover-fit to the canvas) into an `ArtMap`, two small grids in canvas
space: `busy` (local texture and contrast, 0..1) and `subject` (what pulls the eye: colour and brightness that differ from their
surroundings, 0..1). Layout candidates are scored against these maps (`layout/candidates.py`). Deterministic, numpy only; the values
are measurements of the pixels, not a model's opinion."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from poster_agent.spec import Box, PosterSpec

CELL = 16.0  # canvas points per grid cell (the analysis raster is coarser than the compiler's)


@dataclass
class ArtMap:
    busy: np.ndarray  # rows x cols, 0 calm .. 1 busy
    subject: np.ndarray  # rows x cols, 0 background .. 1 salient
    width: float  # canvas size in points
    height: float

    def _cells(self, grid: np.ndarray, box: Box) -> np.ndarray:
        rows, cols = grid.shape
        x0, x1 = int(np.floor(max(box.x, 0) / self.width * cols)), int(np.ceil(min(box.right, self.width) / self.width * cols))
        y0, y1 = int(np.floor(max(box.y, 0) / self.height * rows)), int(np.ceil(min(box.bottom, self.height) / self.height * rows))
        return grid[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]

    def busy_in(self, box: Box) -> float:
        """Mean busyness (texture / contrast) under `box`, 0..1."""
        return float(self._cells(self.busy, box).mean())

    def subject_in(self, box: Box) -> float:
        """Mean saliency under `box`, 0..1: how much of the box covers what the art wants you to look at."""
        return float(self._cells(self.subject, box).mean())

    def subject_centroid(self) -> tuple[float, float] | None:
        """(x, y) of the salient mass in canvas points, or None when the art has no clear subject."""
        w = np.where(self.subject > 0.5, self.subject, 0.0)
        if w.sum() < 0.01 * w.size:
            return None
        rows, cols = w.shape
        ys, xs = np.mgrid[0:rows, 0:cols]
        return (float((xs * w).sum() / w.sum() + 0.5) / cols * self.width, float((ys * w).sum() / w.sum() + 0.5) / rows * self.height)


def _norm(a: np.ndarray, lo_p: float = 5.0, hi_p: float = 95.0) -> np.ndarray:
    lo, hi = np.percentile(a, lo_p), np.percentile(a, hi_p)
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def analyse_image(img: Image.Image, width: float, height: float) -> ArtMap:
    """ArtMap of an image already cropped to the canvas aspect."""
    cols, rows = max(int(width / CELL), 8), max(int(height / CELL), 8)
    small = img.convert("RGB").resize((cols * 2, rows * 2), Image.BILINEAR)
    lum = np.asarray(small.convert("L"), dtype=float) / 255.0
    gy, gx = np.gradient(lum)
    grad = np.hypot(gx, gy)
    local = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)), dtype=float) / 255.0
    contrast = np.abs(lum - local)
    busy = _norm(0.7 * grad + 0.3 * contrast)
    busy = np.asarray(Image.fromarray((busy * 255).astype(np.uint8)).resize((cols, rows), Image.BOX), dtype=float) / 255.0

    arr = np.asarray(small, dtype=float)
    near = np.asarray(small.filter(ImageFilter.GaussianBlur(2)), dtype=float)
    far = np.asarray(small.filter(ImageFilter.GaussianBlur(10)), dtype=float)
    centre_surround = np.linalg.norm(near - far, axis=2)  # a region unlike its surroundings
    rare = np.linalg.norm(arr - arr.reshape(-1, 3).mean(axis=0), axis=2)  # a colour unlike the picture's average
    sal = _norm(0.6 * centre_surround + 0.4 * rare)
    sal = np.asarray(Image.fromarray((sal * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3)).resize((cols, rows), Image.BOX),
                     dtype=float) / 255.0
    return ArtMap(busy=busy, subject=_norm(sal, 20.0, 98.0), width=width, height=height)


CALM_DETAIL = 0.010  # mean luminance gradient (0..1, on a ~270 px wide raster) under text at or below which type sits comfortably on art


def text_detail(img: Image.Image, boxes: list[Box], width: float, height: float) -> float:
    """ABSOLUTE fine detail under text boxes: the mean luminance gradient inside the boxes, on a fixed-size raster so pictures compare.
    (ArtMap.busy is normalised per picture, so it cannot say whether a picture is calm in absolute terms.)"""
    small = img.convert("L").resize((270, max(int(270 * height / width), 8)))
    a = np.asarray(small, dtype=float) / 255.0
    gy, gx = np.gradient(a)
    g = np.hypot(gx, gy)
    rows, cols = g.shape
    vals, weights = [], []
    for b in boxes:
        x0, x1 = int(max(b.x, 0) / width * cols), int(np.ceil(min(b.right, width) / width * cols))
        y0, y1 = int(max(b.y, 0) / height * rows), int(np.ceil(min(b.bottom, height) / height * rows))
        cell = g[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
        vals.append(float(cell.mean()))
        weights.append(b.w * b.h)
    return float(np.average(vals, weights=weights)) if vals else 0.0


def analyse(spec: PosterSpec, root) -> ArtMap | None:
    """ArtMap of the poster's generated or supplied background, or None for plain solid / gradient grounds (nothing to avoid)."""
    from poster_agent.layout.adapt import ground_raster
    raster = ground_raster(spec, root)
    if raster is None:
        return None
    return analyse_image(Image.fromarray(raster), spec.canvas.width, spec.canvas.height)
