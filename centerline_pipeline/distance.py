from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def compute_distance_and_cost(
    mask: np.ndarray,
    k: float = 4.0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Euclidean distance transform and exponential cost volume."""

    # Distance to the nearest background pixel; zero outside the vessel.
    distance = ndi.distance_transform_edt(mask)
    # Outside the mask we clamp to zero to avoid artefacts.
    distance *= mask

    max_d = float(distance.max(initial=0.0))
    if max_d <= 0.0:
        raise ValueError("Distance transform is zero everywhere; check the mask.")

    # Convert to cost by inverting and amplifying differences.
    inverted = (max_d - distance) / max(max_d, eps)
    cost = np.exp(k * inverted)
    cost[~mask] = np.inf

    return distance, cost
