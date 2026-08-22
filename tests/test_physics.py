import math

import torch

from samg.physics import (
    apply_mueller,
    cloude_to_mueller,
    minimum_cloude_eigenvalue,
    mueller_to_cloude,
    probe_norm_bounds,
    project_mueller_psd,
    ridge_operator_from_rays,
    stokes_is_valid,
    tetrahedral_stokes_bank,
)


def test_mueller_cloude_round_trip():
    matrix = torch.randn(32, 4, 4, generator=torch.Generator().manual_seed(7))
    recovered = cloude_to_mueller(mueller_to_cloude(matrix))
    torch.testing.assert_close(recovered, matrix, atol=2e-6, rtol=2e-6)


def test_psd_projection_is_idempotent_and_definition_is_explicit():
    matrix = torch.randn(32, 4, 4, generator=torch.Generator().manual_seed(8))
    projected = project_mueller_psd(matrix)
    second = project_mueller_psd(projected)
    assert float(minimum_cloude_eigenvalue(projected).min()) >= -2e-6
    torch.testing.assert_close(projected, second, atol=4e-6, rtol=4e-6)

    positive_m00 = matrix.clone()
    positive_m00[:, 0, 0] = positive_m00[:, 0, 0].abs() + 0.1
    preserved = project_mueller_psd(positive_m00, preserve_m00=True)
    torch.testing.assert_close(preserved[:, 0, 0], positive_m00[:, 0, 0], atol=2e-5, rtol=2e-5)
    assert float(minimum_cloude_eigenvalue(preserved).min()) >= -3e-6


def test_tetrahedral_bank_is_valid_and_well_conditioned():
    bank = tetrahedral_stokes_bank()
    assert bool(stokes_is_valid(bank).all())
    torch.testing.assert_close(
        torch.linalg.cond(bank), torch.tensor(math.sqrt(3.0)), atol=1e-6, rtol=1e-6
    )


def test_ridge_recovers_full_rank_operator_and_supports_extra_rays():
    generator = torch.Generator().manual_seed(11)
    matrix = torch.randn(3, 4, 4, generator=generator)
    bank = tetrahedral_stokes_bank().expand(3, -1, -1)
    emerging = apply_mueller(matrix, bank)
    recovered = ridge_operator_from_rays(bank, emerging, ridge=0.0)
    torch.testing.assert_close(recovered, matrix, atol=2e-6, rtol=2e-6)

    extra = torch.cat([bank, 0.8 * bank[:, :2]], dim=1)
    recovered_extra = ridge_operator_from_rays(extra, apply_mueller(matrix, extra), ridge=0.0)
    torch.testing.assert_close(recovered_extra, matrix, atol=3e-6, rtol=3e-6)


def test_probe_norm_equivalence_bounds_hold():
    generator = torch.Generator().manual_seed(13)
    bank = torch.randn(7, 4, generator=generator)
    error = torch.randn(5, 4, 4, generator=generator)
    lower, upper = probe_norm_bounds(bank)
    action_norm = torch.linalg.matrix_norm(error @ bank.T)
    operator_norm = torch.linalg.matrix_norm(error)
    assert bool((lower * operator_norm <= action_norm + 1e-6).all())
    assert bool((action_norm <= upper * operator_norm + 1e-6).all())


def test_probe_lower_bound_is_zero_when_bank_does_not_span_stokes_space():
    bank = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    lower, upper = probe_norm_bounds(bank)
    assert lower == 0.0
    assert upper > 0.0
