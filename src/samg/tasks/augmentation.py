"""Interpolation-free D4 augmentation with the matching Stokes-frame change."""

from __future__ import annotations

import torch


def apply_mueller_reference_change(field: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply ``T M T^T`` to a channel-first Mueller field."""
    if field.ndim != 4 or field.shape[1] != 16:
        raise ValueError("Mueller-aware transforms require [B,16,H,W]")
    matrix = field.permute(0, 2, 3, 1).reshape(field.shape[0], field.shape[2], field.shape[3], 4, 4)
    transform = transform.to(device=field.device, dtype=field.dtype)
    transformed = transform @ matrix @ transform.transpose(-2, -1)
    return transformed.reshape(field.shape[0], field.shape[2], field.shape[3], 16).permute(
        0, 3, 1, 2
    )


def polarimetric_d4(
    field: torch.Tensor,
    target: torch.Tensor,
    *,
    quarter_turns: int,
    horizontal_flip: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate/flip the image and transform Mueller coefficients coherently."""
    quarter_turns %= 4
    if quarter_turns:
        field = torch.rot90(field, quarter_turns, dims=(-2, -1))
        target = torch.rot90(target, quarter_turns, dims=(-2, -1))
        theta = field.new_tensor(quarter_turns * torch.pi / 2.0)
        zero = theta.new_zeros(())
        one = theta.new_ones(())
        cosine = torch.cos(2.0 * theta)
        sine = torch.sin(2.0 * theta)
        rotation = torch.stack(
            [
                torch.stack([one, zero, zero, zero]),
                torch.stack([zero, cosine, -sine, zero]),
                torch.stack([zero, sine, cosine, zero]),
                torch.stack([zero, zero, zero, one]),
            ]
        )
        field = apply_mueller_reference_change(field, rotation)
    if horizontal_flip:
        reflection = torch.diag(field.new_tensor([1.0, 1.0, -1.0, -1.0]))
        field = apply_mueller_reference_change(field.flip(-1), reflection)
        target = target.flip(-1)
    return field, target
