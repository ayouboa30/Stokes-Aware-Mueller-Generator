"""Small physically admissible synthetic fixtures for software smoke tests."""

from __future__ import annotations

import torch

from samg.data.contracts import MuellerSplit
from samg.physics.cloude import cloude_to_mueller


def _random_psd_mueller(count: int, generator: torch.Generator) -> torch.Tensor:
    real = torch.randn(count, 4, 4, generator=generator)
    imaginary = torch.randn(count, 4, 4, generator=generator)
    factor = torch.complex(real, imaginary)
    coherency = factor @ factor.mH
    coherency = coherency / coherency.diagonal(dim1=-2, dim2=-1).real.sum(-1)[:, None, None]
    mueller = cloude_to_mueller(coherency).float()
    # Remove the float32 round-trip jitter on the analytically unit trace.
    mueller[:, 0, 0] = 1.0
    return mueller


def make_synthetic_split(
    size: int,
    *,
    split: str,
    seed: int,
    observations_per_unit: int = 8,
) -> MuellerSplit:
    """Generate a deterministic fixture; it is not a scientific benchmark."""
    if size < 1 or observations_per_unit < 1:
        raise ValueError("size and observations_per_unit must be positive")
    generator = torch.Generator().manual_seed(int(seed))
    target = _random_psd_mueller(size, generator)
    patch = target.reshape(size, 16, 1, 1).expand(-1, -1, 5, 5).clone()
    patch += 0.01 * torch.randn(patch.shape, generator=generator)
    patch[:, :, 2, 2] = target.reshape(size, 16)
    position = torch.rand(size, 2, generator=generator)
    spectral = 5.0 * torch.rand(size, 1, generator=generator)
    source_index = torch.zeros(size, dtype=torch.long)
    record_index = torch.arange(size, dtype=torch.long)
    unit_keys = tuple(
        f"synthetic:{split}:unit-{index // observations_per_unit:04d}" for index in range(size)
    )
    return MuellerSplit(
        patch=patch,
        target=target,
        position=position,
        spectral=spectral,
        source_index=source_index,
        record_index=record_index,
        unit_keys=unit_keys,
        split=split,
    ).validate()
