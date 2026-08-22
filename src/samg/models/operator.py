"""Batch-invariant SAMG with an incident-independent encoder.

This architecture retains the central ``S_in -> S_out -> M`` path while using
a deterministic tetrahedral probe for every observation. It is exposed beside,
not as a silent replacement for, the historical four-group architecture.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from samg.physics.stokes import ridge_operator_from_rays, tetrahedral_stokes_bank


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(inplace=True), nn.Linear(width, width)
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.net(value))


class FlatPatchEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 96, width: int = 160, patch_size: int = 5) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.patch_projection = nn.Sequential(
            nn.Linear(16 * self.patch_size * self.patch_size, width),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Sequential(ResidualMLPBlock(width), ResidualMLPBlock(width))
        self.center_projection = nn.Linear(16, width, bias=False)
        self.output = nn.Sequential(nn.Linear(width, hidden_dim), nn.ReLU(inplace=True))

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        expected = (16, self.patch_size, self.patch_size)
        if patch.ndim != 4 or tuple(patch.shape[1:]) != expected:
            raise ValueError(f"Expected [B,16,{self.patch_size},{self.patch_size}]")
        center = patch[:, :, self.patch_size // 2, self.patch_size // 2]
        features = self.patch_projection(patch.flatten(1)) + self.center_projection(center)
        return self.output(self.residual(features))


class IncidentIndependentEncoder(nn.Module):
    def __init__(self, latent_dim: int = 8, hidden_dim: int = 160, patch_emb_dim: int = 96):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(16 + patch_emb_dim + 2 + 1, hidden_dim), nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(ResidualMLPBlock(hidden_dim), ResidualMLPBlock(hidden_dim))
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        matrix_flat: torch.Tensor,
        patch_embedding: torch.Tensor,
        position: torch.Tensor,
        spectral: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(
            self.input(torch.cat([matrix_flat, patch_embedding, position, spectral], dim=-1))
        )
        mu = torch.nan_to_num(self.mu_head(hidden), nan=0.0, posinf=10.0, neginf=-10.0)
        logvar = torch.nan_to_num(self.logvar_head(hidden), nan=0.0, posinf=2.0, neginf=-6.0).clamp(
            -6.0, 2.0
        )
        return mu, logvar


class TetrahedralIncidentSystem(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("bank", tetrahedral_stokes_bank(), persistent=True)

    def forward(self, spectral: torch.Tensor) -> torch.Tensor:
        if spectral.ndim != 2 or spectral.shape[1] != 1:
            raise ValueError(f"Expected [B,1], got {tuple(spectral.shape)}")
        return self.bank.unsqueeze(0).expand(len(spectral), -1, -1)


class OperatorStokesDecoder(nn.Module):
    def __init__(self, latent_dim: int = 8, hidden_dim: int = 160, context_dim: int = 96):
        super().__init__()
        self.context = nn.Sequential(
            nn.Linear(latent_dim + 2 + 1, hidden_dim),
            nn.ReLU(inplace=True),
            ResidualMLPBlock(hidden_dim),
            nn.Linear(hidden_dim, context_dim),
            nn.ReLU(inplace=True),
        )
        self.operator_head = nn.Linear(context_dim, 16)
        self.ray_projection = nn.Sequential(nn.Linear(4, 32), nn.ReLU(inplace=True))
        self.residual_head = nn.Sequential(
            nn.Linear(context_dim + 32, 64), nn.ReLU(inplace=True), nn.Linear(64, 4)
        )
        self.residual_log_scale = nn.Parameter(torch.tensor(-2.5))

    def forward(
        self,
        latent: torch.Tensor,
        position: torch.Tensor,
        spectral: torch.Tensor,
        incident: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = torch.nan_to_num(latent, nan=0.0, posinf=10.0, neginf=-10.0)
        context = self.context(torch.cat([latent, position, spectral], dim=-1))
        operator = self.operator_head(context).view(len(latent), 4, 4)
        coherent = torch.bmm(incident, operator.transpose(1, 2))
        ray_features = self.ray_projection(incident)
        expanded_context = context[:, None, :].expand(-1, 4, -1)
        residual = self.residual_head(torch.cat([expanded_context, ray_features], dim=-1))
        raw = torch.nan_to_num(
            coherent + torch.sigmoid(self.residual_log_scale) * residual,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )
        intensity = F.softplus(raw[..., :1]).clamp(max=1e3)
        direction = raw[..., 1:]
        norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-8)
        polarization = intensity * torch.tanh(norm) * direction / norm
        emerging = torch.cat([intensity, polarization], dim=-1)
        reference = intensity[:, :1, :] + 1e-8
        return emerging / reference, reference


class OperatorPIVAE(nn.Module):
    """Per-observation, incident-independent Stokes-aware VAE."""

    incident_group_count = 4
    incident_conditioning = "per_observation"
    encoder_uses_incident = False

    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dim: int = 160,
        patch_emb_dim: int = 96,
        patch_width: int = 160,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.generator = TetrahedralIncidentSystem()
        self.patch_encoder = FlatPatchEncoder(patch_emb_dim, patch_width)
        self.encoder = IncidentIndependentEncoder(latent_dim, hidden_dim, patch_emb_dim)
        self.decoder = OperatorStokesDecoder(latent_dim, hidden_dim, patch_emb_dim)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def encode(
        self,
        patch: torch.Tensor,
        matrix: torch.Tensor,
        position: torch.Tensor,
        spectral: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = spectral / 5.0
        return self.encoder(
            matrix.reshape(len(matrix), 16), self.patch_encoder(patch), position, normalized
        )

    def forward(
        self,
        patch: torch.Tensor,
        matrix: torch.Tensor,
        position: torch.Tensor,
        spectral: torch.Tensor,
        *,
        pinv_ridge: float = 1e-2,
        sample_latent: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        normalized = spectral / 5.0
        embedding = self.patch_encoder(patch)
        mu, logvar = self.encoder(matrix.reshape(len(matrix), 16), embedding, position, normalized)
        sample = self.training if sample_latent is None else bool(sample_latent)
        latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        incident = self.generator(normalized)
        emerging_normalized, intensity = self.decoder(latent, position, normalized, incident)
        emerging = emerging_normalized * intensity
        reconstructed = ridge_operator_from_rays(
            incident, emerging_normalized, pinv_ridge
        ) * intensity.reshape(len(matrix), 1, 1)
        zero = reconstructed.new_zeros(())
        return {
            "m_hat": reconstructed,
            "s_in": incident,
            "s_out": emerging,
            "z": latent,
            "mu": mu,
            "logvar": logvar,
            "generator_kl": zero,
            "condition_penalty": zero,
            "poincare_penalty": zero,
        }


CPUFourIncidentPIVAE = OperatorPIVAE
