from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
from PIL import Image

_MASK_COLOR = (180, 180, 180)
_PRIMARY_PATH_COLOR = (220, 20, 20)
_START_COLOR = (0, 220, 50)
_END_COLOR = (40, 50, 220)
_PATH_COLOR_CYCLE = [
    (255, 107, 107),
    (76, 110, 245),
    (64, 192, 87),
    (245, 159, 0),
    (121, 80, 242),
    (47, 158, 68),
    (217, 72, 15),
    (18, 184, 134),
    (230, 119, 0),
    (21, 170, 191),
]


def _base_canvas(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool)
    h, w = mask_bool.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[mask_bool] = _MASK_COLOR
    return canvas


def _paint_centerline(canvas: np.ndarray, path: Sequence[Sequence[int]], color: Tuple[int, int, int]) -> None:
    if path is None:
        return
    coords = np.asarray(path, dtype=int)
    if coords.size == 0:
        return
    h, w, _ = canvas.shape
    for r, c in coords:
        r = int(r)
        c = int(c)
        r0 = max(r - 1, 0)
        r1 = min(r + 2, h)
        c0 = max(c - 1, 0)
        c1 = min(c + 2, w)
        canvas[r0:r1, c0:c1] = color


def _plot_point(canvas: np.ndarray, point: Sequence[int] | None, color: Tuple[int, int, int]) -> None:
    if point is None:
        return
    r, c = int(point[0]), int(point[1])
    h, w, _ = canvas.shape
    if 0 <= r < h and 0 <= c < w:
        canvas[r, c] = color


def render_overlay(
    mask: np.ndarray,
    path: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> Image.Image:
    """Create an RGB overlay showing mask and a single centerline."""
    canvas = _base_canvas(mask)
    _paint_centerline(canvas, path, _PRIMARY_PATH_COLOR)
    _plot_point(canvas, start, _START_COLOR)
    _plot_point(canvas, end, _END_COLOR)
    return Image.fromarray(canvas)


def render_multi_overlay(
    mask: np.ndarray,
    paths: Iterable[dict],
    *,
    colors: Sequence[Tuple[int, int, int]] | None = None,
) -> Image.Image:
    """Render an overlay containing all centerlines for a case."""
    canvas = _base_canvas(mask)
    palette = colors or _PATH_COLOR_CYCLE
    for idx, entry in enumerate(paths):
        path = entry.get("path")
        if not path:
            continue
        color = palette[idx % len(palette)]
        _paint_centerline(canvas, path, color)
    return Image.fromarray(canvas)
