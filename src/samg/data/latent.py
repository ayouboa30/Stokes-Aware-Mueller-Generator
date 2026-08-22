"""Leakage-safe contract for spatial latent-field generator splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class LatentFieldSplit:
    fields: torch.Tensor
    unit_keys: tuple[str, ...]
    split: str

    def __len__(self) -> int:
        return len(self.fields)

    def validate(self) -> LatentFieldSplit:
        if len(self.fields) < 1:
            raise ValueError("A latent split must contain at least one field")
        if self.fields.ndim != 4:
            raise ValueError(f"fields must be [N,C,H,W], got {tuple(self.fields.shape)}")
        if len(self.unit_keys) != len(self.fields):
            raise ValueError("unit_keys must contain one stable anonymous key per field")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation or test")
        if not torch.isfinite(self.fields).all():
            raise ValueError("latent fields contain non-finite values")
        return self


def save_latent_split(split: LatentFieldSplit, path: str | Path) -> None:
    split.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "kind": "samg_latent_fields",
            "split": split.split,
            "fields": split.fields.cpu(),
            "unit_keys": list(split.unit_keys),
        },
        destination,
    )


def load_latent_split(path: str | Path, *, expected_split: str | None = None) -> LatentFieldSplit:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("kind") != "samg_latent_fields":
        raise ValueError("Expected a SAMG latent-field split")
    split = LatentFieldSplit(
        fields=torch.as_tensor(payload["fields"]).float(),
        unit_keys=tuple(str(key) for key in payload["unit_keys"]),
        split=str(payload["split"]),
    ).validate()
    if expected_split is not None and split.split != expected_split:
        raise ValueError(f"Expected split {expected_split!r}, found {split.split!r}")
    return split


def validate_latent_disjoint_units(*splits: LatentFieldSplit) -> None:
    seen: dict[str, str] = {}
    for split in splits:
        for key in set(split.validate().unit_keys):
            if key in seen and seen[key] != split.split:
                raise ValueError(
                    f"Biological-unit leakage: anonymous key occurs in {seen[key]} and {split.split}"
                )
            seen[key] = split.split
