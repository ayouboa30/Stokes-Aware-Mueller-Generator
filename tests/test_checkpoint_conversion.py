import pytest
import torch

from tools.convert_trusted_checkpoint import extract_state_dict


def test_extract_tensor_only_state_dict():
    state = {"layer.weight": torch.randn(2, 3), "layer.bias": torch.randn(2)}
    extracted = extract_state_dict({"model": state, "epoch": 4})
    assert set(extracted) == set(state)
    assert all(value.device.type == "cpu" for value in extracted.values())


def test_reject_non_tensor_checkpoint_fields():
    with pytest.raises(ValueError, match="tensors only"):
        extract_state_dict({"model": {"layer.weight": torch.ones(2), "bad": "text"}})
