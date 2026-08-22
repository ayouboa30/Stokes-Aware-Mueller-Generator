"""Stokes actions and stable reconstruction of a Mueller operator."""

from __future__ import annotations

import math

import torch


def tetrahedral_stokes_bank(
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    intensity: float = 1.0,
) -> torch.Tensor:
    """Return four fully polarized tetrahedral Stokes states as rows.

    The bank has singular-value condition number ``sqrt(3)``. The common
    intensity must be strictly positive.
    """
    if intensity <= 0:
        raise ValueError("intensity must be strictly positive")
    direction = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    ) / math.sqrt(3.0)
    ones = torch.ones(4, 1, device=device, dtype=dtype)
    return float(intensity) * torch.cat([ones, direction], dim=-1)


def stokes_is_valid(states: torch.Tensor, *, tolerance: float = 1e-6) -> torch.Tensor:
    """Check the forward Lorentz-cone condition for each Stokes state."""
    if states.shape[-1] != 4:
        raise ValueError(f"Expected (...,4) Stokes states, got {tuple(states.shape)}")
    intensity = states[..., 0]
    polarization = torch.linalg.vector_norm(states[..., 1:], dim=-1)
    return (intensity >= -tolerance) & (polarization <= intensity + tolerance)


def apply_mueller(matrix: torch.Tensor, incident: torch.Tensor) -> torch.Tensor:
    """Apply Mueller matrices to incident Stokes states stored as rows.

    ``matrix`` has shape ``(...,4,4)`` and ``incident`` has shape ``(...,K,4)``.
    Broadcastable leading dimensions are accepted; the result has shape
    ``(...,K,4)``.
    """
    if matrix.shape[-2:] != (4, 4) or incident.shape[-1] != 4:
        raise ValueError("Expected Mueller (...,4,4) and incident (...,K,4)")
    return torch.matmul(matrix, incident.transpose(-1, -2)).transpose(-1, -2)


def ridge_operator_from_rays(
    incident: torch.Tensor,
    emerging: torch.Tensor,
    ridge: float = 1e-2,
) -> torch.Tensor:
    """Fit ``M`` in ``S_out = M S_in`` by batched ridge regression.

    Incident and emerging states are rows with shapes ``(...,K,4)``. The
    returned tensor has shape ``(...,4,4)``. With ``ridge=0`` and a full-rank
    four-state bank, the reconstruction is the exact inverse.
    """
    if incident.shape != emerging.shape or incident.shape[-1] != 4:
        raise ValueError(
            "incident and emerging must have identical shape (...,K,4), got "
            f"{tuple(incident.shape)} and {tuple(emerging.shape)}"
        )
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    incident_columns = incident.transpose(-1, -2)  # (...,4,K)
    emerging_columns = emerging.transpose(-1, -2)  # (...,4,K)
    identity = torch.eye(4, device=incident.device, dtype=incident.dtype)
    gram = incident_columns @ incident_columns.transpose(-1, -2)
    inverse_gram = torch.linalg.solve(gram + float(ridge) * identity, identity)
    return emerging_columns @ incident_columns.transpose(-1, -2) @ inverse_gram


def probe_norm_bounds(incident: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return constants bounding operator error by Stokes-action error.

    For ``E = M - M_hat`` and the column bank ``S = incident.T``:

    ``sigma_min(S) ||E||_F <= ||E S||_F <= sigma_max(S) ||E||_F``.

    The lower constant is positive exactly when the incident bank spans the
    four-dimensional Stokes space.
    """
    if incident.shape[-1] != 4:
        raise ValueError(f"Expected (...,K,4), got {tuple(incident.shape)}")
    singular = torch.linalg.svdvals(incident.transpose(-1, -2))
    rank = torch.linalg.matrix_rank(incident.transpose(-1, -2))
    lower = singular[..., -1]
    lower = torch.where(rank >= 4, lower, torch.zeros_like(lower))
    return lower, singular[..., 0]
