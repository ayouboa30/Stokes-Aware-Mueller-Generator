"""Differentiable multi-bank operator closure losses and diagnostics."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _check_banks(incident: torch.Tensor, response: torch.Tensor) -> None:
    if incident.shape != response.shape or incident.ndim != 4 or incident.shape[-1] != 4:
        raise ValueError("incident and response must both be [batch,groups,rays,4]")


def fit_shared_operator(
    incident: torch.Tensor, response: torch.Tensor, ridge: float = 1e-4
) -> torch.Tensor:
    """Fit one Mueller operator jointly to every incident-bank group."""
    _check_banks(incident, response)
    batch = len(incident)
    columns = incident.reshape(batch, -1, 4).transpose(1, 2)
    outputs = response.reshape(batch, -1, 4).transpose(1, 2)
    identity = torch.eye(4, dtype=columns.dtype, device=columns.device).expand(batch, -1, -1)
    gram = columns @ columns.transpose(1, 2) + float(ridge) * identity
    cross = outputs @ columns.transpose(1, 2)
    return torch.linalg.solve(gram.transpose(1, 2), cross.transpose(1, 2)).transpose(1, 2)


def fit_group_operators(
    incident: torch.Tensor, response: torch.Tensor, ridge: float = 1e-4
) -> torch.Tensor:
    _check_banks(incident, response)
    batch, groups, rays, _ = incident.shape
    fitted = fit_shared_operator(
        incident.reshape(batch * groups, 1, rays, 4),
        response.reshape(batch * groups, 1, rays, 4),
        ridge,
    )
    return fitted.reshape(batch, groups, 4, 4)


def apply_operator(matrix: torch.Tensor, incident: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 3 or matrix.shape[-2:] != (4, 4) or incident.shape[-1] != 4:
        raise ValueError("Expected matrix [B,4,4] and incident [B,...,4]")
    return torch.einsum("bij,b...j->b...i", matrix, incident)


def operator_residual_per_sample(
    incident: torch.Tensor,
    response: torch.Tensor,
    matrix: torch.Tensor | None = None,
    *,
    ridge: float = 1e-4,
    eps: float = 1e-8,
) -> torch.Tensor:
    _check_banks(incident, response)
    fitted = fit_shared_operator(incident, response, ridge) if matrix is None else matrix
    residual = response - apply_operator(fitted, incident)
    numerator = residual.square().sum(dim=(1, 2, 3))
    denominator = response.square().sum(dim=(1, 2, 3)) + float(eps)
    return numerator / denominator


def linearity_loss(
    mixed: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    coefficient_first: torch.Tensor,
    coefficient_second: torch.Tensor,
) -> torch.Tensor:
    while coefficient_first.ndim < first.ndim:
        coefficient_first = coefficient_first.unsqueeze(-1)
        coefficient_second = coefficient_second.unsqueeze(-1)
    return F.mse_loss(mixed, coefficient_first * first + coefficient_second * second)


def operator_loss_bundle(
    incident: torch.Tensor,
    response: torch.Tensor,
    *,
    mixed_response: torch.Tensor | None = None,
    first_response: torch.Tensor | None = None,
    second_response: torch.Tensor | None = None,
    coefficient_first: torch.Tensor | None = None,
    coefficient_second: torch.Tensor | None = None,
    ridge: float = 1e-4,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Return unweighted operator, group and optional linearity losses."""
    shared = fit_shared_operator(incident, response, ridge)
    residual = operator_residual_per_sample(incident, response, shared, ridge=ridge, eps=eps)
    grouped = fit_group_operators(incident, response, ridge)
    group_loss = (grouped - shared[:, None]).square().sum(dim=(-2, -1)).mean()
    output = {
        "op": residual.mean(),
        "group": group_loss,
        "matrix_all": shared,
        "matrix_group": grouped,
        "c_op_per_sample": residual,
    }
    optional = (
        mixed_response,
        first_response,
        second_response,
        coefficient_first,
        coefficient_second,
    )
    if all(value is not None for value in optional):
        output["lin"] = linearity_loss(
            mixed_response,  # type: ignore[arg-type]
            first_response,  # type: ignore[arg-type]
            second_response,  # type: ignore[arg-type]
            coefficient_first,  # type: ignore[arg-type]
            coefficient_second,  # type: ignore[arg-type]
        )
    else:
        output["lin"] = response.new_zeros(())
    return output


@torch.no_grad()
def operator_consistency_metrics(
    incident: torch.Tensor,
    response: torch.Tensor,
    *,
    heldout_incident: torch.Tensor | None = None,
    heldout_response: torch.Tensor | None = None,
    ridge: float = 1e-4,
) -> dict[str, torch.Tensor]:
    fitted = fit_shared_operator(incident, response, ridge)
    consistency = operator_residual_per_sample(incident, response, fitted, ridge=ridge)
    values = {
        "mean": consistency.mean(),
        "median": consistency.median(),
        "q95": torch.quantile(consistency, 0.95),
        "fraction_lt_0.01": (consistency < 0.01).float().mean(),
        "fraction_lt_0.05": (consistency < 0.05).float().mean(),
        "fraction_lt_0.10": (consistency < 0.10).float().mean(),
        "fraction_lt_0.20": (consistency < 0.20).float().mean(),
    }
    if heldout_incident is not None or heldout_response is not None:
        if heldout_incident is None or heldout_response is None:
            raise ValueError("Both heldout tensors are required")
        _check_banks(heldout_incident, heldout_response)
        values["heldout_mean"] = operator_residual_per_sample(
            heldout_incident, heldout_response, fitted, ridge=ridge
        ).mean()
    return values
