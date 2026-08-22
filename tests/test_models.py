import math

import torch

from samg.losses import kl_divergence
from samg.models import DirectMuellerVAE, FourIncidentPIVAE, OperatorPIVAE
from samg.physics import stokes_is_valid


def inputs(batch: int = 8):
    generator = torch.Generator().manual_seed(2044)
    patch = torch.randn(batch, 16, 5, 5, generator=generator)
    matrix = patch[:, :, 2, 2].reshape(batch, 4, 4)
    position = torch.rand(batch, 2, generator=generator)
    spectral = 5.0 * torch.rand(batch, 1, generator=generator)
    return patch, matrix, position, spectral


def test_historical_architecture_shapes_and_parameter_count():
    model = FourIncidentPIVAE().eval()
    output = model(*inputs(), sample_latent=False)
    assert model.parameter_count == 103_712
    assert output["m_hat"].shape == (8, 4, 4)
    assert output["s_in"].shape == (8, 4, 4)
    assert output["s_out"].shape == (8, 4, 4)
    assert output["incident_groups"].shape == (4, 4, 4)
    assert output["z"].shape == (8, 8)
    assert torch.isfinite(output["m_hat"]).all()
    assert bool(stokes_is_valid(output["s_out"]).all())


def test_direct_baseline_shape_and_parameter_count():
    model = DirectMuellerVAE().eval()
    output = model(*inputs(5), sample_latent=False)
    assert model.parameter_count == 109_760
    assert output["m_hat"].shape == (5, 4, 4)
    assert output["z"].shape == (5, 8)
    torch.testing.assert_close(output["kl"], kl_divergence(output["mu"], output["logvar"]))


def test_incident_generator_kl_is_differentiable():
    model = FourIncidentPIVAE().train()
    output = model(*inputs(8), sample_latent=True)
    total_kl = kl_divergence(output["mu"], output["logvar"]) + output["generator_kl"]
    total_kl.backward()
    gradients = [parameter.grad for parameter in model.generator.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_operator_variant_is_batch_context_invariant():
    model = OperatorPIVAE().eval()
    values = inputs(7)
    alone = model(*(value[:1] for value in values), sample_latent=False)
    together = model(*values, sample_latent=False)
    torch.testing.assert_close(alone["m_hat"][0], together["m_hat"][0])
    torch.testing.assert_close(alone["z"][0], together["z"][0])
    condition = torch.linalg.cond(together["s_in"])
    torch.testing.assert_close(
        condition, torch.full_like(condition, math.sqrt(3.0)), rtol=1e-5, atol=1e-5
    )
    assert bool(stokes_is_valid(together["s_out"]).all())


def test_all_models_have_finite_backward_pass():
    for model in (FourIncidentPIVAE(), OperatorPIVAE(), DirectMuellerVAE()):
        model.train()
        output = model(*inputs(4), sample_latent=True)
        loss = output["m_hat"].square().mean() + 1e-4 * output["mu"].square().mean()
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
