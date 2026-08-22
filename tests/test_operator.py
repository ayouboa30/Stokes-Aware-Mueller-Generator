import torch

from samg.operator import (
    apply_operator,
    fit_shared_operator,
    linearity_loss,
    operator_consistency_metrics,
    operator_loss_bundle,
)


def test_exact_shared_operator_recovery():
    generator = torch.Generator().manual_seed(3)
    matrix = torch.randn(7, 4, 4, generator=generator)
    incident = torch.randn(7, 3, 8, 4, generator=generator)
    response = apply_operator(matrix, incident)
    fitted = fit_shared_operator(incident, response, ridge=0.0)
    torch.testing.assert_close(fitted, matrix, atol=2e-5, rtol=2e-5)
    metrics = operator_consistency_metrics(incident, response, ridge=0.0)
    assert metrics["mean"] < 1e-10
    assert metrics["fraction_lt_0.01"] == 1


def test_non_operator_responses_are_detected():
    generator = torch.Generator().manual_seed(4)
    matrix = torch.randn(8, 4, 4, generator=generator)
    incident = torch.randn(8, 3, 8, 4, generator=generator)
    response = apply_operator(matrix, incident) + 0.2 * incident.square()
    losses = operator_loss_bundle(incident, response)
    assert losses["op"] > 1e-4
    assert losses["group"] > 1e-4


def test_linearity_loss_is_zero_for_an_exact_mixture():
    first = torch.randn(6, 4)
    second = torch.randn(6, 4)
    a, b = torch.randn(6), torch.randn(6)
    exact = a[:, None] * first + b[:, None] * second
    assert linearity_loss(exact, first, second, a, b) < 1e-12
