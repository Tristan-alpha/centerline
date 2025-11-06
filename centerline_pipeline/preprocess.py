from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class PreparedCase:
    mask: np.ndarray
    start: Tuple[int, int]
    end: Tuple[int, int]
    dilations: int = 0
    snaps: Dict[str, float] = field(default_factory=dict)


def _clamp_point(shape: Tuple[int, int], point: Tuple[int, int]) -> Tuple[int, int]:
    r, c = point
    max_r, max_c = shape[0] - 1, shape[1] - 1
    return (
        min(max(r, 0), max_r),
        min(max(c, 0), max_c),
    )


def _nearest_foreground(mask: np.ndarray, point: Tuple[int, int]) -> Tuple[Tuple[int, int], float]:
    r, c = point
    vessel_pts = np.column_stack(np.nonzero(mask))
    if vessel_pts.size == 0:
        raise ValueError("Mask does not contain any foreground pixels")
    diff = vessel_pts - np.array([[r, c]])
    dist_sq = np.einsum("ij,ij->i", diff, diff)
    idx = int(dist_sq.argmin())
    nearest = tuple(int(v) for v in vessel_pts[idx])
    distance = float(np.sqrt(dist_sq[idx]))
    return nearest, distance


def _ensure_connected(
    mask: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    *,
    max_iter: int,
    structure: np.ndarray,
) -> Tuple[np.ndarray, int]:
    current = mask.copy()
    for iteration in range(max_iter + 1):
        labeled, _ = ndi.label(current, structure=structure)
        label_start = labeled[start]
        label_end = labeled[end]
        if label_start != 0 and label_start == label_end:
            return current, iteration
        if iteration == max_iter:
            break
        current = ndi.binary_dilation(current, structure=structure)
    raise ValueError("Unable to connect start and end within dilation budget")


def prepare_case(
    mask: np.ndarray,
    start_point: Tuple[int, int],
    end_point: Tuple[int, int],
    snap: bool = True,
    ensure_connected: bool = True,
    max_dilation: int = 10,
) -> PreparedCase:
    """Convert raw mask & annotation points into a usable form.

    Parameters
    ----------
    mask:
        Binary mask, any non-zero values are treated as foreground.
    start_point, end_point:
        Points provided by the annotations JSON. They are interpreted as
        (row, column) indices, matching MATLAB's [y, x] convention.
    snap:
        If True, move annotations that fall outside the mask onto the nearest
        foreground voxel.
    """

    bin_mask = mask.astype(bool)
    if not bin_mask.any():
        raise ValueError("Mask is empty after boolean conversion")

    start = _clamp_point(bin_mask.shape, tuple(int(v) for v in start_point))
    end = _clamp_point(bin_mask.shape, tuple(int(v) for v in end_point))
    snaps: Dict[str, float] = {}
    if snap:
        if not bin_mask[start]:
            nearest, dist = _nearest_foreground(bin_mask, start)
            start = nearest
            snaps["start"] = dist
        if not bin_mask[end]:
            nearest, dist = _nearest_foreground(bin_mask, end)
            end = nearest
            snaps["end"] = dist

    filled = ndi.binary_fill_holes(bin_mask)

    dilations = 0
    mask_connected = filled
    if ensure_connected:
        structure = ndi.generate_binary_structure(rank=2, connectivity=2)
        mask_connected, dilations = _ensure_connected(
            mask_connected, start, end, max_iter=max_dilation, structure=structure
        )

    return PreparedCase(mask=mask_connected, start=start, end=end, dilations=dilations, snaps=snaps)
