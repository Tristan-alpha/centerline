"""Loss functions for scalar field regression."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from torch import Tensor, nn


def _finite_difference(tensor: Tensor, dim: int) -> Tensor:
    """Compute forward finite differences along ``dim`` with zero padding."""
    slicer_positive = [slice(None)] * tensor.ndim
    slicer_negative = [slice(None)] * tensor.ndim
    slicer_positive[dim] = slice(1, None)
    slicer_negative[dim] = slice(None, -1)
    diff = tensor[tuple(slicer_positive)] - tensor[tuple(slicer_negative)]
    pad_shape = list(tensor.shape)
    pad_shape[dim] = 1
    pad = torch.zeros(*pad_shape, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([diff, pad], dim=dim)


def gradient_components(tensor: Tensor) -> List[Tensor]:
    """Return per-axis gradients of ``tensor`` keeping the original shape."""
    grads: List[Tensor] = []
    for axis in range(2, tensor.ndim):
        grads.append(_finite_difference(tensor, axis))
    return grads


def gradient_magnitude(tensor: Tensor, squared: bool = False) -> Tensor:
    """Compute isotropic gradient magnitude of a scalar field tensor."""
    grads = gradient_components(tensor)
    if not grads:
        raise ValueError("Expected spatial dimensions in tensor.")
    squared_mag = sum(component.pow(2.0) for component in grads)
    return squared_mag if squared else torch.sqrt(squared_mag + 1e-12)


def _match_mask_shape(mask: Tensor, reference: Tensor) -> Tensor:
    """Broadcast ``mask`` to match ``reference`` spatial & channel dims."""

    if mask.shape == reference.shape:
        return mask
    if mask.ndim != reference.ndim:
        raise ValueError("mask must have the same number of dimensions as reference.")
    if mask.shape[0] != reference.shape[0]:
        raise ValueError("Batch dimension mismatch between mask and reference.")
    if mask.shape[1] == 1:
        expand_shape = (-1, reference.shape[1], *reference.shape[2:])
        return mask.expand(expand_shape)
    raise ValueError("mask must either match reference shape or have a single channel.")


class ScalarFieldLoss(nn.Module):
    """Composite loss for scalar field regression with mask-aware weighting."""

    def __init__(
        self,
        l1_weight: float = 1.0,
        grad_weight: float = 0.1,
        eikonal_weight: float = 0.0,
        eikonal_target: float = 1.0,
        outside_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.grad_weight = grad_weight
        self.eikonal_weight = eikonal_weight
        self.eikonal_target = eikonal_target
        self.outside_weight = outside_weight

    def forward(self, prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have identical shapes.")

        mask = _match_mask_shape(mask.float(), prediction)
        outside = 1.0 - mask
        eps = 1e-6

        error = (prediction - target).abs()
        inside_loss = (error * mask).sum() / (mask.sum() + eps)
        outside_loss = (error * outside).sum() / (outside.sum() + eps)
        loss = self.l1_weight * (inside_loss + self.outside_weight * outside_loss)

        if self.grad_weight > 0.0:
            pred_grads = gradient_components(prediction)
            target_grads = gradient_components(target)
            grad_terms: Iterable[Tensor] = []
            for pred_grad, target_grad in zip(pred_grads, target_grads):
                grad_error = (pred_grad - target_grad).abs() * mask
                grad_terms.append(grad_error.sum() / (mask.sum() + eps))
            loss = loss + self.grad_weight * sum(grad_terms) / max(len(pred_grads), 1)

        if self.eikonal_weight > 0.0:
            grad_mag = gradient_magnitude(prediction, squared=False)
            eikonal = ((grad_mag - self.eikonal_target).abs() * mask).sum() / (
                mask.sum() + eps
            )
            loss = loss + self.eikonal_weight * eikonal

        return loss
