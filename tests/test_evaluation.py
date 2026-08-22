import torch

from samg.evaluation import evaluate_predictions
from samg.physics import project_mueller_psd


def test_numerical_psd_tolerance_and_positive_margin_are_distinct():
    target = project_mueller_psd(torch.randn(12, 4, 4))
    metrics = evaluate_predictions(
        target,
        target,
        numerical_tolerance=1e-6,
        minimum_eigenvalue=0.01,
    )
    assert metrics["cloude_psd_fraction"] == 1.0
    assert metrics["cloude_margin_fraction"] <= metrics["cloude_psd_fraction"]
    assert metrics["target_cloude_psd_fraction"] == 1.0
    assert metrics["cloude_minimum_eigenvalue_threshold"] == 0.01
