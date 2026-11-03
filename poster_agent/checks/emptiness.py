"""How much of a poster is dead space?

Works on the rendered image alone (plus optional 'designed block' boxes), so it can be calibrated on the designers' real posters.
The poster is cut into ~20pt cells; a cell is *occupied* if it has visible detail (luminance std above a small threshold: text,
photos, patterns, textures) or lies inside a designed block (photo, panel, bar, logo). Occupancy is dilated a little so the
ordinary air around text does not count. What remains is dead space; we report its largest connected region and the longest
fully-empty horizontal band, as fractions of the poster.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from poster_agent.spec import Box

CELL_PT = 20.0
DETAIL_STD = 5.0  # luminance std (0-255) per cell above which a cell has content
DILATE_CELLS = 2  # air allowed around content (~40pt) before a cell counts as empty
MARGIN_PCT = 4.5
BAND_ROW_EMPTY = 0.85  # a row is 'empty' if at least this fraction of its interior cells are empty


@dataclass
class EmptinessReport:
    largest_fraction: float  # largest connected empty region / whole poster
    longest_band_fraction: float  # tallest run of empty rows / poster height
    empty_fraction: float  # all empty cells / whole poster
    regions: list[tuple[Box, float]] = field(default_factory=list)  # (bounding box in pt, area fraction), largest first


def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(m)
            ys, yd = (slice(max(dy, 0), None), slice(0, m.shape[0] - max(dy, 0))) if dy >= 0 else (slice(0, dy), slice(-dy, None))
            xs, xd = (slice(max(dx, 0), None), slice(0, m.shape[1] - max(dx, 0))) if dx >= 0 else (slice(0, dx), slice(-dx, None))
            shifted[ys, xs] = m[yd, xd]
            out |= shifted
    return out


def measure_emptiness(img: Image.Image, scale: float, filled: list[Box] | None = None) -> EmptinessReport:
    """`scale` = pixels per point of `img`; `filled` = designed blocks (in pt) that count as occupied even if smooth."""
    g = np.asarray(img.convert("L"), dtype=np.float32)
    cell = max(int(round(CELL_PT * scale)), 4)
    ny, nx = math.ceil(g.shape[0] / cell), math.ceil(g.shape[1] / cell)
    g = np.pad(g, ((0, ny * cell - g.shape[0]), (0, nx * cell - g.shape[1])), mode="edge")
    detail = g.reshape(ny, cell, nx, cell).std(axis=(1, 3))
    occupied = detail > DETAIL_STD
    for b in filled or []:
        x0, y0 = int(b.x * scale // cell), int(b.y * scale // cell)
        x1, y1 = int(math.ceil(b.right * scale / cell)), int(math.ceil(b.bottom * scale / cell))
        occupied[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
    occupied = _dilate(occupied, DILATE_CELLS)
    m = max(int(math.ceil(MARGIN_PCT / 100.0 * min(nx, ny))), 1)
    interior = np.zeros_like(occupied)
    interior[m:ny - m, m:nx - m] = True
    empty = ~occupied & interior

    # connected empty regions (4-neighbourhood)
    seen = np.zeros_like(empty)
    comps: list[tuple[int, int, int, int, int]] = []  # (cells, x0, y0, x1, y1)
    for sy, sx in zip(*np.nonzero(empty)):
        if seen[sy, sx]:
            continue
        stack, cells, x0, y0, x1, y1 = [(sy, sx)], 0, sx, sy, sx, sy
        seen[sy, sx] = True
        while stack:
            y, x = stack.pop()
            cells += 1
            x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x), max(y1, y)
            for ny2, nx2 in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny2 < ny and 0 <= nx2 < nx and empty[ny2, nx2] and not seen[ny2, nx2]:
                    seen[ny2, nx2] = True
                    stack.append((ny2, nx2))
        comps.append((cells, x0, y0, x1, y1))
    comps.sort(reverse=True)
    total = ny * nx
    regions = [(Box(x=x0 * cell / scale, y=y0 * cell / scale, w=(x1 - x0 + 1) * cell / scale, h=(y1 - y0 + 1) * cell / scale), c / total)
               for c, x0, y0, x1, y1 in comps[:3]]

    # tallest run of rows that are (almost) entirely empty across the interior width
    row_empty = empty[:, m:nx - m].mean(axis=1) >= BAND_ROW_EMPTY
    longest = run = 0
    for r in row_empty:
        run = run + 1 if r else 0
        longest = max(longest, run)
    return EmptinessReport(largest_fraction=(comps[0][0] / total) if comps else 0.0, longest_band_fraction=longest / ny,
                           empty_fraction=float(empty.sum()) / total, regions=regions)
