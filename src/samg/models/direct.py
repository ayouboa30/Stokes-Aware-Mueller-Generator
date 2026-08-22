"""Direct sixteen-coefficient VAE baseline used in matched comparisons."""

from __future__ import annotations

import torch
from torch import nn


class PatchEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        if patch.ndim != 4 or patch.shape[1] != 16:
            raise ValueError(f"Expected [B,16,H,W], got {tuple(patch.shape)}")
        return self.net(patch)


class DirectMuellerVAE(nn.Module):
    """VAE whose decoder directly predicts the sixteen Mueller coefficients."""

    def __init__(self, latent_dim: int = 8, hidden_dim: int = 128, beta_kl: float = 1e-3):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.beta_kl = float(beta_kl)
        self.patch_encoder = PatchEncoder(64)
        self.encoder = nn.Sequential(
            nn.Linear(16 + 64 + 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16),
        )

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
        features = torch.cat(
            [
                matrix.reshape(matrix.shape[0], 16),
                self.patch_encoder(patch),
                position,
                spectral / 5.0,
            ],
            dim=-1,
        )
        hidden = self.encoder(features)
        return self.mu_head(hidden), self.logvar_head(hidden).clamp(-10.0, 5.0)

    def decode(
        self, latent: torch.Tensor, position: torch.Tensor, spectral: torch.Tensor
    ) -> torch.Tensor:
        return self.decoder(torch.cat([latent, position, spectral / 5.0], dim=-1)).reshape(-1, 4, 4)

    def forward(
        self,
        patch: torch.Tensor,
        matrix: torch.Tensor,
        position: torch.Tensor,
        spectral: torch.Tensor,
        *,
        sample_latent: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(patch, matrix, position, spectral)
        sample = self.training if sample_latent is None else bool(sample_latent)
        latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        reconstructed = self.decode(latent, position, spectral)
        kl = -0.5 * torch.sum(1.0 + logvar - mu.square() - logvar.exp()) / len(matrix)
        reconstruction = (reconstructed - matrix).square().mean()
        return {
            "m_hat": reconstructed,
            "mu": mu,
            "logvar": logvar,
            "z": latent,
            "reconstruction": reconstruction,
            "kl": kl,
            "loss": reconstruction + self.beta_kl * kl,
        }


VanillaVAE = DirectMuellerVAE
CNNVanillaVAE = DirectMuellerVAE
