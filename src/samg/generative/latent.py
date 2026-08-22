"""Pixel-autoregressive, diffusion and Flow Matching latent-field models.

The default dimensions match the models trained on eight-channel SAMG latent
fields. VDM and Flow Matching deliberately share the same neural backbone so
their capacity is identical; their objectives and samplers differ.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class MaskedConv2d(nn.Conv2d):
    """Autoregressive convolution with an immutable type-A or type-B mask."""

    def __init__(self, kind: str, *args, **kwargs) -> None:
        if kind not in {"A", "B"}:
            raise ValueError("kind must be 'A' or 'B'")
        super().__init__(*args, **kwargs)
        mask = torch.ones_like(self.weight)
        _, _, height, width = self.weight.shape
        mask[:, :, height // 2, width // 2 + (kind == "B") :] = 0.0
        mask[:, :, height // 2 + 1 :] = 0.0
        self.register_buffer("mask", mask)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            value,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class PixelCNN(nn.Module):
    def __init__(self, width: int = 96, depth: int = 6, channels: int = 8) -> None:
        super().__init__()
        if width < 1 or depth < 0 or channels < 1:
            raise ValueError("width/channels must be positive and depth non-negative")
        self.channels = int(channels)
        layers: list[nn.Module] = [
            MaskedConv2d("A", channels, width, 7, padding=3),
            nn.GELU(),
        ]
        for _ in range(depth):
            layers.extend([MaskedConv2d("B", width, width, 3, padding=1), nn.GELU()])
        layers.append(nn.Conv2d(width, 2 * channels, 1))
        self.net = nn.Sequential(*layers)

    def loss(self, field: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.net(field).chunk(2, dim=1)
        logvar = logvar.clamp(-7.0, 4.0)
        return 0.5 * ((field - mean).square() / logvar.exp() + logvar).mean()

    @torch.no_grad()
    def sample(
        self,
        count: int,
        shape: tuple[int, int],
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        field = torch.zeros(count, self.channels, *shape, device=device)
        for row in range(shape[0]):
            for column in range(shape[1]):
                mean, logvar = self.net(field).chunk(2, dim=1)
                noise = torch.randn(count, self.channels, device=device, generator=generator)
                field[:, :, row, column] = (
                    mean[:, :, row, column]
                    + (0.5 * logvar[:, :, row, column].clamp(-7.0, 4.0)).exp() * noise
                )
        return field


class PixelRNN(nn.Module):
    def __init__(self, width: int = 96, channels: int = 8) -> None:
        super().__init__()
        if width < 1 or channels < 1:
            raise ValueError("width and channels must be positive")
        self.channels = int(channels)
        self.input = MaskedConv2d("A", channels, width, 7, padding=3)
        self.cell = nn.GRU(width, width, num_layers=2, batch_first=True)
        self.output = nn.Conv2d(width, 2 * channels, 1)

    def _forward(self, field: torch.Tensor) -> torch.Tensor:
        features = F.gelu(self.input(field))
        batch, width, height, columns = features.shape
        rows = features.permute(0, 2, 3, 1).reshape(batch * height, columns, width)
        processed, _ = self.cell(rows)
        processed = processed.reshape(batch, height, columns, width).permute(0, 3, 1, 2)
        return self.output(processed)

    def loss(self, field: torch.Tensor) -> torch.Tensor:
        mean, logvar = self._forward(field).chunk(2, dim=1)
        logvar = logvar.clamp(-7.0, 4.0)
        return 0.5 * ((field - mean).square() / logvar.exp() + logvar).mean()

    @torch.no_grad()
    def sample(
        self,
        count: int,
        shape: tuple[int, int],
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        field = torch.zeros(count, self.channels, *shape, device=device)
        for row in range(shape[0]):
            for column in range(shape[1]):
                mean, logvar = self._forward(field).chunk(2, dim=1)
                noise = torch.randn(count, self.channels, device=device, generator=generator)
                field[:, :, row, column] = (
                    mean[:, :, row, column]
                    + (0.5 * logvar[:, :, row, column].clamp(-7.0, 4.0)).exp() * noise
                )
        return field


class TimeConditionedUNet(nn.Module):
    def __init__(self, width: int = 96, channels: int = 8) -> None:
        super().__init__()
        if width < 1 or channels < 1:
            raise ValueError("width and channels must be positive")
        self.channels = int(channels)
        self.time = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, width))
        self.down = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
        )
        self.up = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, channels, 3, padding=1),
        )

    def forward(self, field: torch.Tensor, moment: torch.Tensor) -> torch.Tensor:
        embedding = self.time(moment.reshape(-1, 1))[:, :, None, None]
        hidden = self.down(field) + embedding
        hidden = self.middle(hidden) + hidden
        hidden = F.interpolate(hidden, size=field.shape[-2:], mode="nearest")
        return self.up(hidden)


class LatentVDM(nn.Module):
    """Cosine-schedule latent diffusion model with noise prediction."""

    def __init__(self, steps: int = 200, width: int = 96, channels: int = 8) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError("steps must be positive")
        self.channels = int(channels)
        self.net = TimeConditionedUNet(width=width, channels=channels)
        self.steps = int(steps)
        moments = torch.linspace(0.0, 1.0, self.steps + 1)
        alphas = torch.cos((moments + 0.008) / 1.008 * math.pi / 2).square()
        self.register_buffer("bar", (alphas / alphas[0]).clamp(1e-5, 1.0))

    def loss(self, field: torch.Tensor) -> torch.Tensor:
        index = torch.randint(1, self.steps + 1, (len(field),), device=field.device)
        alpha = self.bar[index][:, None, None, None]
        noise = torch.randn_like(field)
        noisy = alpha.sqrt() * field + (1.0 - alpha).sqrt() * noise
        return F.mse_loss(self.net(noisy, index.float() / self.steps), noise)

    @torch.no_grad()
    def sample(
        self,
        count: int,
        shape: tuple[int, int],
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        field = torch.randn(count, self.channels, *shape, device=device, generator=generator)
        for index in range(self.steps, 0, -1):
            moment = torch.full((count,), index / self.steps, device=device)
            predicted_noise = self.net(field, moment)
            alpha, previous = self.bar[index], self.bar[index - 1]
            estimate = ((field - (1.0 - alpha).sqrt() * predicted_noise) / alpha.sqrt()).clamp(
                -4.0, 4.0
            )
            if index > 1:
                noise = torch.randn(field.shape, device=device, generator=generator)
                field = previous.sqrt() * estimate + (1.0 - previous).sqrt() * noise
            else:
                field = estimate
        return field


class FlowMatching(nn.Module):
    """Rectified conditional Flow Matching on latent fields."""

    def __init__(self, steps: int = 100, width: int = 96, channels: int = 8) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError("steps must be positive")
        self.channels = int(channels)
        self.net = TimeConditionedUNet(width=width, channels=channels)
        self.steps = int(steps)

    def loss(self, field: torch.Tensor) -> torch.Tensor:
        moment = torch.rand(len(field), device=field.device)
        noise = torch.randn_like(field)
        path = moment[:, None, None, None] * field + (1.0 - moment[:, None, None, None]) * noise
        return F.mse_loss(self.net(path, moment), field - noise)

    @torch.no_grad()
    def sample(
        self,
        count: int,
        shape: tuple[int, int],
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        field = torch.randn(count, self.channels, *shape, device=device, generator=generator)
        step = 1.0 / self.steps
        for index in range(self.steps):
            moment = torch.full((count,), index * step, device=device)
            field = field + step * self.net(field, moment)
        return field


LatentDiffusion = LatentVDM


def build_latent_generator(name: str, **kwargs) -> nn.Module:
    families = {
        "pixelcnn": PixelCNN,
        "pixelrnn": PixelRNN,
        "vdm": LatentVDM,
        "flow": FlowMatching,
    }
    try:
        return families[name.lower()](**kwargs)
    except KeyError as error:
        raise ValueError(f"Unknown family {name!r}; choose from {sorted(families)}") from error
