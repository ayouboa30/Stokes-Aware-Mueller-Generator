import torch

from samg.tasks import MultiTaskModel


def test_multitask_shapes_and_shared_capacity():
    heads = {"tissue": ("segmentation", 4), "diagnosis": ("classification", 2)}
    model = MultiTaskModel(heads, in_channels=8, width=8)
    latent = torch.randn(3, 8, 16, 16)
    assert model(latent, "tissue").shape == (3, 4, 16, 16)
    assert model(latent, "diagnosis").shape == (3, 2)
    single = MultiTaskModel({"tissue": ("segmentation", 4)}, in_channels=8, width=8)
    assert model.trunk_parameters() == single.trunk_parameters()
    assert model.head_parameters("tissue") == single.head_parameters("tissue")
