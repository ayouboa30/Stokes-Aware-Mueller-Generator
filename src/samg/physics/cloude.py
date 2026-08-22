"""Canonical Mueller--Cloude transforms for the ``(I,Q,U,V)`` convention.

The normalization is

``H(M) = 1/4 sum_ij M_ij (sigma_i kron conjugate(sigma_j))``

and therefore ``trace(H) == M00``.
"""

from __future__ import annotations

import torch


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype in (torch.float64, torch.complex128) else torch.complex64


def stokes_pauli_basis(
    *,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    return torch.tensor(
        [
            [[1, 0], [0, 1]],
            [[1, 0], [0, -1]],
            [[0, 1], [1, 0]],
            [[0, -1j], [1j, 0]],
        ],
        dtype=dtype,
        device=device,
    )


def cloude_basis(
    *,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    sigma = stokes_pauli_basis(dtype=dtype, device=device)
    return torch.einsum("iac,jbd->ijabcd", sigma, sigma.conj()).reshape(4, 4, 4, 4)


def as_mueller_matrices(values: torch.Tensor) -> torch.Tensor:
    """Convert ``(...,4,4)``, ``(...,16)`` or ``(B,16,H,W)`` to matrices."""
    if values.shape[-2:] == (4, 4):
        return values
    if values.shape[-1:] == (16,):
        return values.reshape(*values.shape[:-1], 4, 4)
    if values.ndim == 4 and values.shape[1] == 16:
        return values.permute(0, 2, 3, 1).reshape(
            values.shape[0], values.shape[2], values.shape[3], 4, 4
        )
    raise ValueError(
        f"Unsupported Mueller layout {tuple(values.shape)}; expected (...,4,4), "
        "(...,16) or (B,16,H,W)"
    )


def mueller_to_cloude(mueller: torch.Tensor) -> torch.Tensor:
    matrix = as_mueller_matrices(mueller)
    if torch.is_complex(matrix):
        raise ValueError("Mueller matrices must be real-valued")
    dtype = _complex_dtype(matrix.dtype)
    basis = cloude_basis(dtype=dtype, device=matrix.device)
    coherency = 0.25 * torch.einsum("...ij,ijab->...ab", matrix.to(dtype), basis)
    return 0.5 * (coherency + coherency.mH)


def cloude_to_mueller(coherency: torch.Tensor) -> torch.Tensor:
    if coherency.shape[-2:] != (4, 4):
        raise ValueError(f"Expected Cloude matrices (...,4,4), got {tuple(coherency.shape)}")
    basis = cloude_basis(dtype=coherency.dtype, device=coherency.device)
    return torch.real(torch.einsum("...ab,ijba->...ij", coherency, basis))


def cloude_eigenvalues(mueller: torch.Tensor) -> torch.Tensor:
    coherency = mueller_to_cloude(mueller)
    return torch.linalg.eigvalsh(0.5 * (coherency + coherency.mH)).real


def minimum_cloude_eigenvalue(mueller: torch.Tensor) -> torch.Tensor:
    return cloude_eigenvalues(mueller)[..., 0]


def negative_spectral_mass(mueller: torch.Tensor) -> torch.Tensor:
    return -cloude_eigenvalues(mueller).clamp_max(0.0).sum(dim=-1)


def check_psd(
    coherency: torch.Tensor, *, tolerance: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    if coherency.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (...,4,4), got {tuple(coherency.shape)}")
    hermitian = 0.5 * (coherency + coherency.mH)
    eigenvalues = torch.linalg.eigvalsh(hermitian).real
    return eigenvalues, eigenvalues[..., 0] < -float(tolerance)


def project_cloude_psd(coherency: torch.Tensor, *, min_eigenvalue: float = 0.0) -> torch.Tensor:
    """Project Hermitian matrices onto the PSD cone by eigenvalue clipping."""
    if min_eigenvalue < 0:
        raise ValueError("min_eigenvalue must be non-negative for a PSD projection")
    if coherency.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (...,4,4), got {tuple(coherency.shape)}")
    hermitian = 0.5 * (coherency + coherency.mH)
    eigenvalues, eigenvectors = torch.linalg.eigh(hermitian)
    clipped = eigenvalues.clamp_min(float(min_eigenvalue))
    projected = (eigenvectors * clipped.unsqueeze(-2)) @ eigenvectors.mH
    return 0.5 * (projected + projected.mH)


def project_mueller_psd(
    mueller: torch.Tensor,
    *,
    min_eigenvalue: float = 0.0,
    preserve_m00: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Project through ``H(M)`` while preserving the input layout.

    By default this is the Euclidean spectral projection onto the full PSD cone
    in Cloude space; its trace, and therefore ``M00``, may change. With
    ``preserve_m00=True`` a positive scalar renormalization is applied *after*
    the cone projection. That option is not claimed to be the Euclidean
    projection onto a fixed-trace intersection.
    """
    original_shape = mueller.shape
    channel_first = mueller.ndim == 4 and mueller.shape[1] == 16
    flat = mueller.shape[-1:] == (16,)
    matrices = as_mueller_matrices(mueller)
    projected = cloude_to_mueller(
        project_cloude_psd(mueller_to_cloude(matrices), min_eigenvalue=min_eigenvalue)
    ).to(mueller.dtype)
    if preserve_m00:
        target_m00 = matrices[..., :1, :1]
        if torch.any(target_m00 <= 0):
            raise ValueError("preserve_m00 requires strictly positive input M00")
        current_m00 = projected[..., :1, :1]
        projected = projected * (target_m00 / current_m00.clamp_min(eps))
    if channel_first:
        return projected.reshape(
            original_shape[0], original_shape[2], original_shape[3], 16
        ).permute(0, 3, 1, 2)
    if flat:
        return projected.reshape(original_shape)
    return projected
