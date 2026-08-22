"""Validated, private-data-free contract for SAMG tensor splits."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

SCHEMA_VERSION = 1


@dataclass
class MuellerSplit:
    """One already-partitioned set of Mueller observations.

    ``unit_keys`` are stable anonymous keys such as ``source:random-hash``.
    They must not be split-local integer indices and must not contain real
    patient or specimen identifiers.
    """

    patch: torch.Tensor
    target: torch.Tensor
    position: torch.Tensor
    spectral: torch.Tensor
    source_index: torch.Tensor
    record_index: torch.Tensor
    unit_keys: tuple[str, ...]
    split: str

    def __len__(self) -> int:
        return len(self.target)

    def validate(self) -> MuellerSplit:
        count = len(self)
        if count < 1:
            raise ValueError("A split must contain at least one observation")
        expected = {
            "patch": (count, 16, 5, 5),
            "target": (count, 4, 4),
            "position": (count, 2),
            "spectral": (count, 1),
            "source_index": (count,),
            "record_index": (count,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        if len(self.unit_keys) != count:
            raise ValueError("unit_keys must contain one stable anonymous key per observation")
        if any(not isinstance(key, str) or not key.strip() for key in self.unit_keys):
            raise ValueError("Every unit key must be a non-empty string")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation or test")
        if torch.any((self.position < -1e-6) | (self.position > 1.0 + 1e-6)):
            raise ValueError("position must be normalized to [0,1] using train-only statistics")
        if torch.any((self.spectral < -1e-6) | (self.spectral > 5.0 + 1e-6)):
            raise ValueError("spectral must use the documented dimensionless [0,5] scale")
        return self

    def tensors(self, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, name).to(device)
            for name in ("patch", "target", "position", "spectral")
        }

    def subset(self, indices: torch.Tensor | list[int]) -> MuellerSplit:
        index = torch.as_tensor(indices, dtype=torch.long)
        return MuellerSplit(
            patch=self.patch[index],
            target=self.target[index],
            position=self.position[index],
            spectral=self.spectral[index],
            source_index=self.source_index[index],
            record_index=self.record_index[index],
            unit_keys=tuple(self.unit_keys[int(i)] for i in index),
            split=self.split,
        ).validate()


def save_split(split: MuellerSplit, path: str | Path) -> None:
    split.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "split": split.split,
        "patch": split.patch.cpu(),
        "target": split.target.cpu(),
        "position": split.position.cpu(),
        "spectral": split.spectral.cpu(),
        "source_index": split.source_index.cpu(),
        "record_index": split.record_index.cpu(),
        "unit_keys": list(split.unit_keys),
    }
    torch.save(payload, destination)


def load_split(path: str | Path, *, expected_split: str | None = None) -> MuellerSplit:
    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{source} does not follow SAMG tensor schema v{SCHEMA_VERSION}")
    split_name = str(payload.get("split", ""))
    if expected_split is not None and split_name != expected_split:
        raise ValueError(f"Expected split {expected_split!r}, found {split_name!r}")
    split = MuellerSplit(
        patch=torch.as_tensor(payload["patch"]).float(),
        target=torch.as_tensor(payload["target"]).float(),
        position=torch.as_tensor(payload["position"]).float(),
        spectral=torch.as_tensor(payload["spectral"]).float(),
        source_index=torch.as_tensor(payload["source_index"]).long(),
        record_index=torch.as_tensor(payload["record_index"]).long(),
        unit_keys=tuple(str(key) for key in payload["unit_keys"]),
        split=split_name,
    )
    return split.validate()


def validate_disjoint_units(splits: Iterable[MuellerSplit]) -> None:
    """Reject overlap of stable ``(source, unit)`` keys between partitions."""
    seen: dict[str, str] = {}
    for split in splits:
        split.validate()
        for key in set(split.unit_keys):
            previous = seen.get(key)
            if previous is not None and previous != split.split:
                raise ValueError(
                    f"Biological-unit leakage: anonymous key occurs in {previous} and {split.split}"
                )
            seen[key] = split.split
