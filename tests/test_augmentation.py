import torch

from samg.physics import cloude_eigenvalues
from samg.tasks import polarimetric_d4


def test_d4_preserves_shape_mask_labels_and_cloude_spectrum():
    generator = torch.Generator().manual_seed(21)
    field = torch.randn(2, 16, 8, 8, generator=generator)
    mask = torch.randint(0, 4, (2, 8, 8), generator=generator)
    matrices = field.permute(0, 2, 3, 1).reshape(2, 8, 8, 4, 4)
    before = cloude_eigenvalues(matrices).reshape(-1, 4).sort(dim=0).values
    transformed, transformed_mask = polarimetric_d4(
        field, mask, quarter_turns=1, horizontal_flip=True
    )
    transformed_matrices = transformed.permute(0, 2, 3, 1).reshape(2, 8, 8, 4, 4)
    after = cloude_eigenvalues(transformed_matrices).reshape(-1, 4).sort(dim=0).values
    assert transformed.shape == field.shape
    assert transformed_mask.shape == mask.shape
    assert set(transformed_mask.unique().tolist()) == set(mask.unique().tolist())
    torch.testing.assert_close(after, before, atol=4e-6, rtol=4e-6)
