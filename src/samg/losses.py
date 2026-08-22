"""Reconstruction and regularization losses shared by the training commands."""

from __future__ import annotations

import torch

from samg.physics.stokes import apply_mueller


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=-1).mean()


def stokes_action_mse(
    emerging: torch.Tensor, target_matrix: torch.Tensor, incident: torch.Tensor
) -> torch.Tensor:
    return (emerging - apply_mueller(target_matrix, incident)).square().mean()


def channel_scale(
    training_target: torch.Tensor, *, tolerance: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit per-channel scales on training matrices only.

    Constant channels receive scale one instead of an unstable near-zero
    denominator.
    """
    deviation = training_target.std(dim=0, unbiased=False)
    degenerate = deviation <= tolerance
    return torch.where(degenerate, torch.ones_like(deviation), deviation), degenerate


def normalized_matrix_mse(
    prediction: torch.Tensor, target: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return ((prediction - target) / scale).square().mean()
