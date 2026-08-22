"""Historical conditional CNN PI-VAE and its four-bank training variant.

The parameter names and layer dimensions intentionally match the clean
research extraction so that its architecture-only state dictionaries remain
compatible. Incident states are represented as rows of a ``4 x 4`` tensor.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from samg.physics.stokes import ridge_operator_from_rays


class PatchEncoder(nn.Module):
    """Encode a local Mueller patch ``[B,16,H,W]`` into 64 features."""

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


class GlobalStokesGeneratorConditional(nn.Module):
    """Learn one admissible four-state incident bank from a spectral value.

    The historical architecture conditions the bank on the mean value of the
    supplied group. Intensities are positive and each polarization vector lies
    on the Poincare sphere.
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 32),
        )

    def forward(
        self, spectral: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if spectral.ndim != 2 or spectral.shape[1] != 1:
            raise ValueError(f"Expected spectral condition [B,1], got {tuple(spectral.shape)}")
        parameters = self.net(spectral.mean(dim=0, keepdim=True))
        s0_mu = parameters[:, 0:4]
        s0_logvar = parameters[:, 4:8].clamp(min=-6.0, max=2.0)
        direction_mu = parameters[:, 8:20].view(-1, 4, 3)
        direction_logvar = parameters[:, 20:32].view(-1, 4, 3).clamp(min=-6.0, max=2.0)

        if self.training:
            s0_raw = s0_mu + torch.exp(0.5 * s0_logvar) * torch.randn_like(s0_mu)
            direction = direction_mu + torch.exp(0.5 * direction_logvar) * torch.randn_like(
                direction_mu
            )
        else:
            s0_raw = s0_mu
            direction = direction_mu

        intensity = F.softplus(s0_raw)
        direction = F.normalize(direction, p=2, dim=-1)
        incident = torch.cat(
            [intensity.unsqueeze(-1), intensity.unsqueeze(-1) * direction], dim=-1
        ).squeeze(0)

        kl_s0 = -0.5 * torch.sum(1 + s0_logvar - s0_mu.square() - s0_logvar.exp())
        kl_direction = -0.5 * torch.sum(
            1 + direction_logvar - direction_mu.square() - direction_logvar.exp()
        )
        zero = incident.new_zeros(())
        return incident, kl_s0, zero, kl_direction


class LocalMuellerEncoderConditionalCNN(nn.Module):
    """Encode the centre, patch, position, spectral value and incident bank."""

    def __init__(self, latent_dim: int = 8, hidden_dim: int = 128, patch_emb_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16 + patch_emb_dim + 2 + 1 + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        matrix_flat: torch.Tensor,
        patch_embedding: torch.Tensor,
        position: torch.Tensor,
        spectral: torch.Tensor,
        incident: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = matrix_flat.shape[0]
        incident_flat = (
            incident.reshape(16).expand(batch_size, 16)
            if incident.ndim == 2
            else incident.reshape(batch_size, 16)
        )
        features = self.net(
            torch.cat([matrix_flat, patch_embedding, position, spectral, incident_flat], dim=-1)
        )
        mu = torch.nan_to_num(self.mu_head(features), nan=0.0, posinf=10.0, neginf=-10.0)
        logvar = torch.nan_to_num(self.logvar_head(features), nan=0.0, posinf=2.0, neginf=-6.0)
        return mu, logvar


class LocalMuellerDecoderConditionalCNN(nn.Module):
    """Decode four emerging Stokes states inside the Lorentz cone."""

    def __init__(self, latent_dim: int = 8, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 2 + 1 + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16),
        )

    def forward(
        self,
        latent: torch.Tensor,
        position: torch.Tensor,
        spectral: torch.Tensor,
        incident: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = latent.shape[0]
        latent = torch.nan_to_num(latent, nan=0.0, posinf=10.0, neginf=-10.0)
        incident_flat = (
            incident.reshape(16).expand(batch_size, 16)
            if incident.ndim == 2
            else incident.reshape(batch_size, 16)
        )
        raw = torch.nan_to_num(
            self.net(torch.cat([latent, position, spectral, incident_flat], dim=-1)).view(
                batch_size, 4, 4
            ),
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )
        intensity = F.softplus(raw[..., :1]).clamp(max=1e3)
        polarization_raw = raw[..., 1:]
        norm = torch.sqrt(polarization_raw.square().sum(dim=-1, keepdim=True) + 1e-12)
        polarization = intensity * torch.tanh(norm) * polarization_raw / (norm + 1e-8)
        emerging = torch.cat([intensity, polarization], dim=-1)
        reference_intensity = intensity[:, 0:1, :] + 1e-8
        return emerging / reference_intensity, reference_intensity


class ConditionalCNNPIVAE(nn.Module):
    """Historical conditional CNN PI-VAE with one incident bank per batch."""

    def __init__(
        self,
        latent_dim: int = 8,
        loss_type: str = "stokes",
        normalize_stokes_loss: bool = False,
        beta_kl: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.beta_kl = float(beta_kl)
        self.loss_type = loss_type
        self.normalize_stokes_loss = bool(normalize_stokes_loss)
        self.generator = GlobalStokesGeneratorConditional()
        self.patch_encoder = PatchEncoder(hidden_dim=64)
        self.encoder = LocalMuellerEncoderConditionalCNN(latent_dim=latent_dim, patch_emb_dim=64)
        self.decoder = LocalMuellerDecoderConditionalCNN(latent_dim=latent_dim)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def cond_penalty(states: torch.Tensor) -> torch.Tensor:
        states = torch.nan_to_num(states, nan=0.0, posinf=1e6, neginf=-1e6)
        condition = torch.nan_to_num(
            torch.linalg.cond(states).clamp(min=1.0), nan=1e6, posinf=1e6, neginf=1e6
        )
        return (condition - math.sqrt(3.0)).square().mean()

    cond_penalty_sout = cond_penalty

    @staticmethod
    def poincare_penalty(states: torch.Tensor) -> torch.Tensor:
        intensity_squared = states[..., 0].square().clamp(min=1e-8)
        polarization_squared = states[..., 1:].square().sum(dim=-1)
        return F.relu(1000.0 * polarization_squared / intensity_squared - 1000.0).mean()

    @staticmethod
    def _broadcast_incident(incident: torch.Tensor, batch_size: int) -> torch.Tensor:
        if incident.ndim == 2:
            return incident.unsqueeze(0).expand(batch_size, -1, -1)
        if incident.ndim == 3 and incident.shape[0] in (1, batch_size):
            return incident.expand(batch_size, -1, -1)
        raise ValueError(
            f"Unexpected incident shape {tuple(incident.shape)} for batch size {batch_size}"
        )

    def _sample_latent(
        self, mu: torch.Tensor, logvar: torch.Tensor, sample_latent: bool | None
    ) -> torch.Tensor:
        sample = self.training if sample_latent is None else bool(sample_latent)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu

    def encode(
        self,
        patch: torch.Tensor,
        matrix: torch.Tensor,
        position: torch.Tensor,
        spectral: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = patch.shape[0]
        spectral_normalized = spectral / 5.0
        incident, _, _, _ = self.generator(spectral_normalized)
        incident = self._broadcast_incident(incident, batch_size)
        return self.encoder(
            matrix.reshape(batch_size, 16),
            self.patch_encoder(patch),
            position,
            spectral_normalized,
            incident,
        )

    def _incident_for_batch(
        self, spectral_normalized: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        incident, kl_s0, kl_radius, kl_direction = self.generator(spectral_normalized)
        incident = self._broadcast_incident(incident, len(spectral_normalized))
        return incident, kl_s0 + kl_radius + kl_direction, {}

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
        batch_size = patch.shape[0]
        spectral_normalized = spectral / 5.0
        incident, generator_kl, extras = self._incident_for_batch(spectral_normalized)
        mu, logvar = self.encoder(
            matrix.reshape(batch_size, 16),
            self.patch_encoder(patch),
            position,
            spectral_normalized,
            incident,
        )
        latent = self._sample_latent(mu, logvar, sample_latent)
        emerging_normalized, intensity_scale = self.decoder(
            latent, position, spectral_normalized, incident
        )
        emerging = emerging_normalized * intensity_scale
        reconstructed = ridge_operator_from_rays(incident, emerging_normalized, pinv_ridge)
        reconstructed = reconstructed * intensity_scale.reshape(batch_size, 1, 1)
        return {
            "m_hat": reconstructed,
            "s_in": incident,
            "s_out": emerging,
            "z": latent,
            "mu": mu,
            "logvar": logvar,
            "generator_kl": generator_kl,
            **extras,
        }


class FourIncidentPIVAE(ConditionalCNNPIVAE):
    """Historical variant using four learned incident banks across a batch.

    A batch is split into at most four contiguous groups. Each group receives a
    bank conditioned on its mean spectral value. This reproduces the historical
    campaign semantics; use :class:`OperatorPIVAE` when per-observation,
    batch-invariant incident states are required.
    """

    incident_group_count = 4

    def _generate_grouped_incident(
        self, spectral_normalized: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(spectral_normalized) == 0:
            raise ValueError("An empty batch has no incident system")
        group_count = min(self.incident_group_count, len(spectral_normalized))
        spectral_groups = torch.tensor_split(spectral_normalized, group_count, dim=0)
        banks: list[torch.Tensor] = []
        kl_terms: list[torch.Tensor] = []
        sizes: list[int] = []
        for group in spectral_groups:
            bank, kl_s0, kl_radius, kl_direction = self.generator(group)
            banks.append(bank)
            kl_terms.append(kl_s0 + kl_radius + kl_direction)
            sizes.append(len(group))
        grouped = torch.stack(banks, dim=0)
        size_tensor = torch.tensor(sizes, device=spectral_normalized.device)
        per_sample = torch.repeat_interleave(grouped, size_tensor, dim=0)
        return per_sample, grouped, torch.stack(kl_terms).mean(), size_tensor

    def _incident_for_batch(
        self, spectral_normalized: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        incident, groups, generator_kl, sizes = self._generate_grouped_incident(spectral_normalized)
        return incident, generator_kl, {"incident_groups": groups, "group_sizes": sizes}

    def encode(
        self,
        patch: torch.Tensor,
        matrix: torch.Tensor,
        position: torch.Tensor,
        spectral: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = patch.shape[0]
        spectral_normalized = spectral / 5.0
        incident, _, _, _ = self._generate_grouped_incident(spectral_normalized)
        return self.encoder(
            matrix.reshape(batch_size, 16),
            self.patch_encoder(patch),
            position,
            spectral_normalized,
            incident,
        )


CNNConditionalPIVAE = ConditionalCNNPIVAE
