"""Aggregate fidelity and physical-admissibility metrics."""

from __future__ import annotations

import torch

from samg.physics.cloude import cloude_eigenvalues
from samg.physics.stokes import stokes_is_valid


def summarize_values(values: torch.Tensor, prefix: str) -> dict[str, float]:
    flat = values.detach().float().reshape(-1).cpu()
    quantiles = torch.quantile(flat, torch.tensor([0.05, 0.50, 0.95]))
    return {
        f"{prefix}_mean": float(flat.mean()),
        f"{prefix}_median": float(quantiles[1]),
        f"{prefix}_q05": float(quantiles[0]),
        f"{prefix}_q95": float(quantiles[2]),
    }


def evaluate_predictions(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    emerging: torch.Tensor | None = None,
    numerical_tolerance: float = 1e-6,
    minimum_eigenvalue: float = 0.0,
) -> dict[str, float | list[float]]:
    if prediction.shape != target.shape or prediction.shape[-2:] != (4, 4):
        raise ValueError("prediction and target must have identical shape (...,4,4)")
    error = prediction - target
    eigenvalues = cloude_eigenvalues(prediction)
    minimum = eigenvalues[..., 0]
    target_eigenvalues = cloude_eigenvalues(target)
    target_minimum = target_eigenvalues[..., 0]
    if numerical_tolerance < 0 or minimum_eigenvalue < 0:
        raise ValueError("PSD tolerance and minimum eigenvalue must be non-negative")
    quantiles = torch.quantile(minimum.reshape(-1).float(), torch.tensor([0.05, 0.95]))
    metrics: dict[str, float | list[float]] = {
        "mse": float(error.square().mean()),
        "mae": float(error.abs().mean()),
        "cloude_psd_fraction": float((minimum >= -numerical_tolerance).float().mean()),
        "cloude_margin_fraction": float(
            (minimum >= minimum_eigenvalue - numerical_tolerance).float().mean()
        ),
        "cloude_minimum_eigenvalue_threshold": float(minimum_eigenvalue),
        "cloude_lambda_min_mean": float(minimum.mean()),
        "cloude_lambda_min_median": float(minimum.median()),
        "cloude_lambda_min_q05": float(quantiles[0]),
        "cloude_lambda_min_q95": float(quantiles[1]),
        "cloude_negative_mass_mean": float((-eigenvalues.clamp_max(0.0).sum(-1)).mean()),
        "target_cloude_psd_fraction": float(
            (target_minimum >= -numerical_tolerance).float().mean()
        ),
        "target_cloude_margin_fraction": float(
            (target_minimum >= minimum_eigenvalue - numerical_tolerance).float().mean()
        ),
        "mse_per_channel": error.square().reshape(-1, 16).mean(0).tolist(),
    }
    if emerging is not None:
        valid = stokes_is_valid(emerging)
        metrics["stokes_ray_fraction"] = float(valid.float().mean())
        metrics["stokes_all_rays_fraction"] = float(valid.all(dim=-1).float().mean())
    return metrics
