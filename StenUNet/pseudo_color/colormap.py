"""Utilities to turn scalar fields into pseudo color maps."""

from __future__ import annotations

import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import torch
from torch import Tensor

_STENOSIS_COLORMAP = LinearSegmentedColormap.from_list(
    "stenosis_rdylbu",
    [
        (0.0, "#b10026"),  # critical narrowing - saturated red
        (0.3, "#f46d43"),  # severe narrowing - orange
        (0.5, "#fee08b"),  # moderate - yellow
        (0.75, "#66c2a5"),  # mild - aqua
        (1.0, "#3288bd"),  # normal vessel - blue
    ],
)

register_cmap = getattr(cm, "register_cmap", None)
if callable(register_cmap):
    try:
        register_cmap(name=_STENOSIS_COLORMAP.name, cmap=_STENOSIS_COLORMAP)
    except ValueError:
        # Matplotlib raises if the colormap is registered multiple times during reloads.
        pass


def _prepare_scalar(scalar_field: Tensor, clamp: bool) -> Tensor:
    scalar = scalar_field.detach()
    if clamp:
        scalar = scalar.clamp(0.0, 1.0)
    return scalar


def apply_colormap(
    scalar_field: Tensor,
    cmap_name: str = _STENOSIS_COLORMAP.name,
    clamp: bool = True,
) -> Tensor:
    """Map a scalar field to RGB pseudo color space using matplotlib colormaps.

    Args:
        scalar_field: Tensor of shape (N, 1, H, W) or (1, H, W) in [0, 1].
        cmap_name: Name of the matplotlib colormap to use. Defaults to a custom
            red-yellow-blue map that highlights severe stenosis with warm colors.
        clamp: Clamp the scalar field to [0, 1] before mapping.

    Returns:
        Tensor of shape (N, 3, H, W) or (3, H, W) matching the input batch layout.
    """
    scalar = _prepare_scalar(scalar_field, clamp)
    device, dtype = scalar.device, scalar.dtype

    squeeze_batch = scalar.ndim == 3
    if squeeze_batch:
        scalar = scalar.unsqueeze(0)

    if scalar.shape[1] != 1:
        raise ValueError("Expected scalar_field with a single channel.")

    cmap = cm.get_cmap(cmap_name)
    np_values = scalar.squeeze(1).cpu().numpy()
    colored = cmap(np_values)[..., :3]
    tensor = torch.from_numpy(colored).permute(0, 3, 1, 2).to(device=device, dtype=dtype)

    if squeeze_batch:
        tensor = tensor.squeeze(0)
    return tensor
