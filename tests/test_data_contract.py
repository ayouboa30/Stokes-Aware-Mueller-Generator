from pathlib import Path

import pytest
import torch

from samg.data import (
    LatentFieldSplit,
    load_latent_split,
    load_split,
    make_synthetic_split,
    save_latent_split,
    save_split,
    validate_disjoint_units,
)


def test_synthetic_split_round_trip(tmp_path: Path):
    split = make_synthetic_split(16, split="train", seed=3)
    path = tmp_path / "train.pt"
    save_split(split, path)
    recovered = load_split(path, expected_split="train")
    assert len(recovered) == 16
    assert recovered.unit_keys == split.unit_keys
    assert recovered.target.shape == (16, 4, 4)


def test_global_unit_keys_detect_leakage_even_when_indices_are_local():
    training = make_synthetic_split(8, split="train", seed=1)
    validation = make_synthetic_split(8, split="validation", seed=2)
    validate_disjoint_units([training, validation])
    validation.unit_keys = (training.unit_keys[0],) + validation.unit_keys[1:]
    with pytest.raises(ValueError, match="leakage"):
        validate_disjoint_units([training, validation])


def test_latent_field_contract_round_trip(tmp_path: Path):
    split = LatentFieldSplit(
        fields=torch.randn(5, 8, 4, 4),
        unit_keys=tuple(f"synthetic:unit-{index}" for index in range(5)),
        split="train",
    )
    path = tmp_path / "latent.pt"
    save_latent_split(split, path)
    recovered = load_latent_split(path, expected_split="train")
    torch.testing.assert_close(recovered.fields, split.fields)
    assert recovered.unit_keys == split.unit_keys
