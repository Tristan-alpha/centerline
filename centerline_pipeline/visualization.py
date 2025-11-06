from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def render_overlay(
    mask: np.ndarray,
    path: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> Image.Image:
    """Create an RGB overlay showing mask and centerline."""
    h, w = mask.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    vessel = mask.astype(bool)
    canvas[vessel] = (180, 180, 180)

    # Paint centerline pixels in red. Thicken by painting a 3x3 neighborhood.
    for r, c in path:
        r = int(r)
        c = int(c)
        r0 = max(r - 1, 0)
        r1 = min(r + 2, h)
        c0 = max(c - 1, 0)
        c1 = min(c + 2, w)
        canvas[r0:r1, c0:c1] = (220, 20, 20)

    sr, sc = start
    er, ec = end
    canvas[sr, sc] = (0, 220, 50)
    canvas[er, ec] = (40, 50, 220)

    return Image.fromarray(canvas)
