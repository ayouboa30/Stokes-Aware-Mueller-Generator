"""Shared convolutional trunk with task-specific lightweight heads."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    return next(group for group in range(min(8, channels), 0, -1) if channels % group == 0)


class _Stage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = _groups(out_channels)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.body(value)


class SharedTrunk(nn.Module):
    """Fully convolutional U-shaped trunk for a spatial latent field."""

    def __init__(self, in_channels: int = 8, width: int = 32) -> None:
        super().__init__()
        self.enter = _Stage(in_channels, width)
        self.down1 = _Stage(width, 2 * width)
        self.down2 = _Stage(2 * width, 4 * width)
        self.up1 = _Stage(6 * width, 2 * width)
        self.up2 = _Stage(3 * width, width)
        self.out_channels = width

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first = self.enter(value)
        second = self.down1(F.avg_pool2d(first, 2))
        third = self.down2(F.avg_pool2d(second, 2))
        up1 = self.up1(
            torch.cat(
                [
                    F.interpolate(
                        third, size=second.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    second,
                ],
                dim=1,
            )
        )
        return self.up2(
            torch.cat(
                [
                    F.interpolate(up1, size=first.shape[-2:], mode="bilinear", align_corners=False),
                    first,
                ],
                dim=1,
            )
        )


class SegmentationHead(nn.Module):
    def __init__(self, width: int, classes: int) -> None:
        super().__init__()
        self.body = nn.Conv2d(width, classes, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.body(features)


class ClassificationHead(nn.Module):
    def __init__(self, width: int, classes: int) -> None:
        super().__init__()
        self.body = nn.Linear(width, classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.body(features.mean(dim=(-2, -1)))


class MultiTaskModel(nn.Module):
    """One shared trunk and one minimal head per declared task.

    ``heads`` maps a task name to ``("segmentation" | "classification", classes)``.
    A single-task baseline can use the same class with one entry, preserving the
    trunk and head capacity exactly.
    """

    def __init__(
        self,
        heads: dict[str, tuple[str, int]],
        *,
        in_channels: int = 8,
        width: int = 32,
    ) -> None:
        super().__init__()
        if not heads:
            raise ValueError("At least one task head is required")
        self.trunk = SharedTrunk(in_channels=in_channels, width=width)
        built: dict[str, nn.Module] = {}
        for name, (kind, classes) in heads.items():
            if classes < 1:
                raise ValueError(f"Task {name!r} must have at least one output")
            if kind == "segmentation":
                built[name] = SegmentationHead(width, classes)
            elif kind == "classification":
                built[name] = ClassificationHead(width, classes)
            else:
                raise ValueError(f"Unknown task kind {kind!r}")
        self.heads = nn.ModuleDict(built)

    def forward(self, latent_field: torch.Tensor, task: str) -> torch.Tensor:
        if task not in self.heads:
            raise KeyError(f"Unknown task {task!r}; choose from {list(self.heads)}")
        return self.heads[task](self.trunk(latent_field))

    def trunk_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.trunk.parameters())

    def head_parameters(self, task: str) -> int:
        return sum(parameter.numel() for parameter in self.heads[task].parameters())
