from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CenterlineMetrics:
    radii: np.ndarray
    diameters: np.ndarray
    stenosis_curve: np.ndarray
    max_stenosis: float
    max_stenosis_index: int
    path_length: float


def sample_radii(distance: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Sample the distance transform along the centerline."""
    rows, cols = path[:, 0], path[:, 1]
    return distance[rows, cols]


def compute_metrics(distance: np.ndarray, path: np.ndarray) -> CenterlineMetrics:
    radii = sample_radii(distance, path)
    diameters = 2.0 * radii

    prefix_max = np.maximum.accumulate(radii)
    safe = np.where(prefix_max > 0, prefix_max, 1.0)
    stenosis_curve = 1.0 - radii / safe
    max_idx = int(stenosis_curve.argmax())
    max_stenosis = float(stenosis_curve[max_idx])

    diffs = np.diff(path.astype(float), axis=0)
    steps = np.linalg.norm(diffs, axis=1)
    path_length = float(steps.sum())

    return CenterlineMetrics(
        radii=radii,
        diameters=diameters,
        stenosis_curve=stenosis_curve,
        max_stenosis=max_stenosis,
        max_stenosis_index=max_idx,
        path_length=path_length,
    )
