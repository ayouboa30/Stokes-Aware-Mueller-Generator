import torch

from samg.cli.train_generator import _validation_loss
from samg.generative import FlowMatching, LatentVDM, PixelCNN, PixelRNN


def test_all_latent_generator_losses_are_finite_and_differentiable():
    field = torch.randn(2, 8, 8, 8, generator=torch.Generator().manual_seed(5))
    models = (
        PixelCNN(width=8, depth=1),
        PixelRNN(width=8),
        LatentVDM(steps=2, width=8),
        FlowMatching(steps=2, width=8),
    )
    for model in models:
        loss = model.loss(field)
        assert torch.isfinite(loss)
        loss.backward()
        assert any(parameter.grad is not None for parameter in model.parameters())


def test_all_latent_generators_sample_finite_fields():
    models = (
        PixelCNN(width=8, depth=1),
        PixelRNN(width=8),
        LatentVDM(steps=2, width=8),
        FlowMatching(steps=2, width=8),
    )
    for index, model in enumerate(models):
        generator = torch.Generator().manual_seed(40 + index)
        sampled = model.sample(2, (4, 4), torch.device("cpu"), generator)
        assert sampled.shape == (2, 8, 4, 4)
        assert torch.isfinite(sampled).all()


def test_masked_convolution_does_not_mutate_weights():
    model = PixelCNN(width=8, depth=1)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    model.loss(torch.randn(2, 8, 4, 4)).backward()
    for name, value in model.state_dict().items():
        if name.endswith("weight") or name.endswith("mask"):
            torch.testing.assert_close(value, before[name])


def test_stochastic_validation_is_repeatable_and_preserves_training_rng():
    model = FlowMatching(steps=2, width=8)
    fields = torch.randn(4, 8, 8, 8)
    torch.manual_seed(123)
    before = torch.random.get_rng_state()
    first = _validation_loss(model, fields, 2, seed=77)
    after = torch.random.get_rng_state()
    second = _validation_loss(model, fields, 2, seed=77)
    assert first == second
    torch.testing.assert_close(after, before)
