"""Train-only affine normalization for Mueller channels."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _broadcast(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    values = values.to(device=reference.device, dtype=reference.dtype)
    if reference.shape[-2:] == (4, 4):
        return values.reshape(*([1] * (reference.ndim - 2)), 4, 4)
    if reference.ndim == 4 and reference.shape[1] == 16:
        return values.reshape(1, 16, 1, 1)
    if reference.shape[-1:] == (16,):
        return values.reshape(*([1] * (reference.ndim - 1)), 16)
    raise ValueError(f"Unsupported Mueller layout {tuple(reference.shape)}")


@dataclass
class MuellerNormalizer:
    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-6

    def __post_init__(self) -> None:
        self.mean = torch.as_tensor(self.mean, dtype=torch.float32).reshape(16)
        self.std = torch.as_tensor(self.std, dtype=torch.float32).reshape(16)
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all():
            raise ValueError("Normalization statistics must be finite")
        if torch.any(self.std <= 0):
            raise ValueError("All standard deviations must be positive")

    @classmethod
    def fit(cls, training_matrices: torch.Tensor, *, eps: float = 1e-6) -> MuellerNormalizer:
        if training_matrices.shape[-2:] == (4, 4):
            flat = training_matrices.reshape(-1, 16)
        elif training_matrices.shape[-1:] == (16,):
            flat = training_matrices.reshape(-1, 16)
        elif training_matrices.ndim == 4 and training_matrices.shape[1] == 16:
            flat = training_matrices.permute(0, 2, 3, 1).reshape(-1, 16)
        else:
            raise ValueError(f"Unsupported layout {tuple(training_matrices.shape)}")
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=False).clamp_min(eps)
        return cls(mean, std, eps)

    def normalize(self, physical: torch.Tensor) -> torch.Tensor:
        return (physical - _broadcast(self.mean, physical)) / _broadcast(self.std, physical)

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized * _broadcast(self.std, normalized) + _broadcast(self.mean, normalized)

    def state_dict(self) -> dict[str, object]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "eps": self.eps}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> MuellerNormalizer:
        return cls(state["mean"], state["std"], float(state.get("eps", 1e-6)))
